import os
import requests
import subprocess
import json
import time
import cv2
import base64
import uuid
import shutil
from pathlib import Path
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# ================= 配置区域 =================
load_dotenv()
API_KEY = os.getenv("DASHSCOPE_API_KEY", "")

# 模型配置
MODEL_QWEN_PLUS = "qwen3.6-plus"
MODEL_FUN_ASR = "fun-asr"
MODEL_OCR = "qwen3.5-flash"

# 业务参数
SIZE_THRESHOLD_MB = 50 
DANMU_CROP = (25, 1200, 600, 340)
WATCH_CROP = (25, 130,  300,  70)
LIKE_CROP  = (710, 1575, 100, 100)
FRAME_INTERVAL  = 1
FRAME_WORKERS = 20  
DEDUP_WINDOW_SECONDS = 3 

# 确保临时文件夹存在
os.makedirs("temp_storage", exist_ok=True)

# 内存数据库，用于保存任务进度（部署生产环境时可换成 Redis）
TASK_DB = {}

# ================= FastAPI 实例 =================
app = FastAPI(title="直播视频分析 API")

# 允许前端跨域请求 (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 实际部署后可以改成 Lovable 的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= 工具函数：重试与时间处理 =================
def retry_logic(max_retries=3, delay=2):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"[失败] {func.__name__} 最终失败: {e}")
                        raise
                    time.sleep(delay * (attempt + 1))
        return wrapper
    return decorator

def time_to_sec(time_str):
    if not time_str or time_str == "N/A": return 0
    parts = str(time_str).split(':')
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    return 0

def clip_video(input_path, start_sec, end_sec, output_path):
    cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-ss', str(start_sec), '-to', str(end_sec),
        '-c:v', 'copy', '-c:a', 'copy', output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path

# ================= 分析核心类 (已适配 Task_ID 隔离文件) =================

class Part1GlobalAnalyzer:
    @retry_logic(max_retries=2)
    def compress_video_dynamic(self, input_path, task_id):
        file_size_mb = os.path.getsize(input_path) / (1024 * 1024)
        if file_size_mb <= SIZE_THRESHOLD_MB: return input_path
        output_path = f"temp_storage/{task_id}_p1_ready.mp4"
        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-vf', "scale='if(gt(iw,ih),min(720,iw),-2):if(gt(ih,iw),min(720,ih),-2)'",
            '-vcodec', 'libx264', '-crf', '30', '-preset', 'fast',
            '-acodec', 'aac', '-ac', '1', '-ar', '16000',
            '-movflags', '+faststart', output_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return output_path

    @retry_logic(max_retries=3)
    def upload_to_oss(self, file_path, model_name):
        url = "https://dashscope.aliyuncs.com/api/v1/uploads"
        headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
        policy_resp = requests.get(url, headers=headers, params={"action": "getPolicy", "model": model_name}, timeout=15)
        policy = policy_resp.json()['data']
        file_name = Path(file_path).name
        key = f"{policy['upload_dir']}/{file_name}"
        with open(file_path, 'rb') as file:
            files = {
                'OSSAccessKeyId': (None, policy['oss_access_key_id']),
                'Signature': (None, policy['signature']),
                'policy': (None, policy['policy']),
                'x-oss-object-acl': (None, policy['x_oss_object_acl']),
                'x-oss-forbid-overwrite': (None, policy['x_oss_forbid_overwrite']),
                'key': (None, key),
                'success_action_status': (None, '200'),
                'file': (file_name, file)
            }
            response = requests.post(policy['upload_host'], files=files, timeout=60)
            if response.status_code != 200: raise Exception("OSS上传失败")
        return f"oss://{key}"

    @retry_logic(max_retries=2)
    def analyze_video(self, oss_url):
        client = OpenAI(api_key=API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", timeout=120.0)
        # 省略长Prompt以节省篇幅，这里使用你原来的Prompt
        prompt = """
# Role:
你是一位资深的电商直播运营专家和数据分析师，擅长通过视频分析主播的表现、产品逻辑及直播间的转化效率。

# Task:
请对上传的直播视频切片进行深度复盘。你需要精准提取视频中的关键信息，并按照我要求的 JSON 格式进行结构化输出。

# Analysis Dimensions:
-视频摘要：概括直播切片的核心主题及调性。
-主播表现：评估主播的风格（亲和力/专业度/激情）、销售状态及全程的情绪曲线（是否有断档或高潮）。
-章节速览：根据介绍的产品对音视频内容进行章节划分与总结，记录准确的开始与结束时间点。
-直播亮点：复盘吸引人的瞬间、成功的逼单手段或优秀的互动话术。
-改进建议：基于视频中的不足（如节奏拖沓、光线问题、回复不及时等）提出实操性建议。

# Output Format Rules:
1. 输出必须且仅包含一个符合 RFC 8259 规范的 JSON 对象。
2. 不要包含任何多余的开场白或解释文字。
3. section_info 数组中的每个对象代表一个章节。
4. 如果视频中某项信息未提及，请填入 "N/A"；决不允许编造信息。

# Output Format
{
  "video_summary": "100-200字，全局概述",
  "host_analysis": {
    "style": "主播风格描述",
    "selling_status": "整体销售表现评价",
    "emotional_shifts": "情绪变化路径及临场反应",
    "interaction_frequency": "互动频率及质量评价"
  },
  "section_info": [
    {
      "start_time": "MM:SS",
      "end_time": "MM:SS",
      "title": "一句话概括章节标题"
    }
  ],
  "live_highlights": "100-150字，总结做的好的地方",
  "improvement_suggestions": "100-150字，针对性的优化策略"
}    

# Important Note:
必须基于上传的直播视频切片回答，所有提取的信息必须源自视频的语音、字幕或画面。如果视频中没有提到某个维度，该字段必须填写 "N/A"，禁止瞎答，禁止编造虚假数字。
"""
        
        completion = client.chat.completions.create(
            model=MODEL_QWEN_PLUS,
            messages=[{"role": "user", "content": [{"type": "video_url", "video_url": {"url": oss_url}, "fps": 0.2}, {"type": "text", "text": prompt}]}],
            extra_body={"enable_thinking": False, "response_format": {"type": "json_object"}},
            extra_headers={"X-DashScope-OssResourceResolve": "enable"}
        )
        return json.loads(completion.choices[0].message.content)

    def run(self, video_path, task_id):
        working_file = self.compress_video_dynamic(video_path, task_id)
        oss_url = self.upload_to_oss(working_file, MODEL_QWEN_PLUS)
        result = self.analyze_video(oss_url)
        if working_file != video_path and os.path.exists(working_file): os.remove(working_file)
        return result

class Part2ASR:
    @retry_logic()
    def convert_to_mp3(self, video_path, task_id):
        mp3_path = f"temp_storage/{task_id}_audio.mp3"
        cmd = ['ffmpeg', '-i', video_path, '-y', '-vn', '-acodec', 'libmp3lame', '-ac', '1', '-q:a', '2', mp3_path]
        subprocess.run(cmd, check=True, capture_output=True)
        return mp3_path

    @retry_logic()
    def submit_and_poll(self, oss_url):
        submit_url = "https://dashscope.aliyuncs.com/api/v1/services/audio/asr/transcription"
        headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json", "X-DashScope-Async": "enable", "X-DashScope-OssResourceResolve": "enable"}
        payload = {"model": MODEL_FUN_ASR, "input": {"file_urls": [oss_url]}, "parameters": {"diarization_enabled": True, "enable_words": False, "channel_id": [0]}}
        
        resp_json = requests.post(submit_url, headers=headers, json=payload, timeout=15).json()
        task_id = resp_json['output']['task_id']
        
        while True:
            status_resp = requests.get(f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}", headers={"Authorization": f"Bearer {API_KEY}"}, timeout=15).json()
            task_status = status_resp.get('output', {}).get('task_status')
            if task_status == 'SUCCEEDED':
                final_url = status_resp['output']['results'][0]['transcription_url']
                data = requests.get(final_url, timeout=15).json()
                formatted_output = []
                for tr in data.get('transcripts', []):
                    for s in tr.get('sentences', []):
                        formatted_output.append({
                            "start_sec": s['begin_time'] / 1000.0,
                            "time_str": f"{s['begin_time']//60000:02d}:{(s['begin_time']%60000)//1000:02d}",
                            "speaker": s.get('speaker_id', '0'),
                            "text": s['text']
                        })
                return formatted_output
            elif task_status in ['FAILED', 'CANCELED']:
                raise Exception("ASR任务失败")
            time.sleep(5)

    def run(self, video_path, task_id):
        mp3_path = self.convert_to_mp3(video_path, task_id)
        oss_url = Part1GlobalAnalyzer().upload_to_oss(mp3_path, MODEL_FUN_ASR)
        result = self.submit_and_poll(oss_url)
        if os.path.exists(mp3_path): os.remove(mp3_path)
        return result

class Part3Danmu:
    def __init__(self):
        self.client = OpenAI(api_key=API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", timeout=30.0)

    def extract_frames(self, video_path, task_id):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_second = 0
        tasks = []
        
        while True:
            frame_id = int(current_second * fps)
            if frame_id >= total_frames: break
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            if not ret: break

            h_max, w_max = frame.shape[:2]
            paths = {}
            for key, (x, y, w, h) in [("danmu", DANMU_CROP), ("watch", WATCH_CROP), ("like", LIKE_CROP)]:
                cropped = frame[y:min(y+h, h_max), x:min(x+w, w_max)]
                path = f"temp_storage/{task_id}_{key}_{current_second}.jpg"
                cv2.imwrite(path, cropped)
                paths[key] = path

            tasks.append((current_second, paths))
            current_second += FRAME_INTERVAL
        cap.release()
        return tasks

    @retry_logic(max_retries=2)
    def safe_api_call(self, prompt, img_path):
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode('utf-8')
        response = self.client.chat.completions.create(
            model=MODEL_OCR,
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}],
            extra_body={'enable_thinking': False}, response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)

    def process_frame(self, ts, paths):
        # 保持你原来的 Prompt
        danmu_prompt = f"""
你是一个直播弹幕OCR提取助手。请从截图中提取第 {ts} 秒画面里所有可见的完整弹幕条目。

【视觉结构说明】：
每条弹幕在画面中从左到右依次由三部分组成：
- [用户等级]：彩色背景的方框标签，内含数字或文字等级，位于该条弹幕的最左侧
- [用户名]：紧随等级标签之后的彩色文字（非白色），与弹幕内容之间有明显间隔
- [弹幕内容]：用户名之后的纯白色文字，直到该行结束

【提取规则】：
1. 必须严格按照 [用户等级] → [用户名] → [弹幕内容] 的顺序提取，不得调换或合并。
2. [用户名] 与 [弹幕内容] 之间以明显空格分隔：空格左侧为用户名（彩色），右侧为内容（白色）。禁止将用户名混入弹幕内容，或将弹幕内容误认为用户名。
3. 三个字段均为必填项，任何一项为空的条目必须丢弃，不得输出。
4. 仅提取画面中完整可见的弹幕；被截断、被遮挡、无法确认完整性的条目一律丢弃。

【输出要求】：
1. 只输出一个合法的 JSON 对象，格式严格如下，不得包含任何解释文字、注释或 Markdown 格式：
   {{"danmu_list": [ {{"timestamp": {ts}, "user_level": "...", "user_name": "...", "content": "..."}} ]}}
2. 若画面中没有可识别的弹幕，返回：{{"danmu_list": []}}
"""
        try:
            danmu_res = self.safe_api_call(danmu_prompt, paths["danmu"])
            return {"timestamp": ts, "danmu_list": danmu_res.get("danmu_list", [])}
        finally:
            for p in paths.values():
                if os.path.exists(p): os.remove(p)

    def run(self, video_path, task_id):
        tasks = self.extract_frames(video_path, task_id)
        all_danmu = []
        with ThreadPoolExecutor(max_workers=FRAME_WORKERS) as executor:
            future_to_paths = {executor.submit(self.process_frame, ts, paths): ts for ts, paths in tasks}
            for future in as_completed(future_to_paths):
                try:
                    res = future.result(timeout=45)
                    all_danmu.extend(res.get("danmu_list", []))
                except Exception:
                    pass
        return sorted(all_danmu, key=lambda x: x.get('timestamp', 0))

class Part4SectionDetails:
    def run(self, video_path, section_info, task_id):
        results = []
        uploader = Part1GlobalAnalyzer() 
        for idx, sec in enumerate(section_info):
            sec_start_str = sec.get("start_time", "00:00")
            sec_end_str = sec.get("end_time", "00:00")
            start_sec = time_to_sec(sec_start_str)
            end_sec = time_to_sec(sec_end_str)
            if end_sec <= start_sec: continue
            
            clip_name = f"temp_storage/{task_id}_clip_sec{idx}.mp4"
            clip_video(video_path, start_sec, end_sec, clip_name)
            
            try:
                oss_url = uploader.upload_to_oss(clip_name, MODEL_QWEN_PLUS)
                prompt = f"""
# Role:
你是一位资深的电商直播运营专家和数据分析师，擅长通过视频分析主播的表现、产品逻辑及直播间的转化效率。

# Context:
当前你正在分析的视频切片，在原完整直播视频中的时间段是【{sec_start_str} 至 {sec_end_str}】。

# Task:
请对上传的直播视频切片进行深度复盘。精准提取视频中的关键信息，并按照我要求的 JSON 格式进行结构化输出。

# Analysis Dimensions:
-视频摘要：概括直播片段的核心主题及调性。
-产品详情：识别视频中出现的每一个独立产品，并详细记录其物理属性和营销逻辑等信息。

# Output Format Rules:
1. 输出必须且仅包含一个符合 RFC 8259 规范的 JSON 对象。
2. 不要包含任何多余的开场白或解释文字。
3. 如果视频中某项信息未提及，请填入 "N/A"；决不允许编造信息。
4. 必须在顶层返回 start_time 和 end_time，严格原样填入 {sec_start_str} 和 {sec_end_str}。

# Output Format
{{
  "video_summary": "50-100字的切片概述",
  "start_time": "{sec_start_str}",
  "end_time": "{sec_end_str}",
  "products": [
    {{
      "product_name": "产品全称",
      "brand": "品牌",
      "specs": {{
        "color": "颜色",
        "material": "材质/成分"
      }},
      "marketing_attributes": {{
        "style": "产品风格",
        "occasion": "适用场合",
        "function": "核心功能/卖点",
        "visual_elements": "视觉记忆点"
      }},
      "sales_logic": {{
        "seed_logic": "种草话术/为什么要买",
        "closing_strategy": "逼单策略（如：限时、限额、低价诱惑）",
        "deal_price": "直播间到手价",
        "discount_intensity": "优惠力度计算及评价"
      }},
      "presentation_data": {{
        "duration_seconds": "讲解时长（秒）",
        "stock_feedback": "视频中反馈的销售/库存情况",
        "audience_concerns": "该产品讲解期间观众最集中的疑问点"
      }}
    }}
  ]
}}

# Note [Important]:
必须基于上传的直播视频切片回答，所有提取的信息必须源自视频的语音、字幕或画面。
""" 
                
                client = OpenAI(api_key=API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", timeout=120.0)
                completion = client.chat.completions.create(
                    model=MODEL_QWEN_PLUS,
                    messages=[{"role": "user", "content": [{"type": "video_url", "video_url": {"url": oss_url}, "fps": 0.5}, {"type": "text", "text": prompt}]}],
                    extra_body={"enable_thinking": False, "response_format": {"type": "json_object"}},
                    extra_headers={"X-DashScope-OssResourceResolve": "enable"}
                )
                sec_detail = json.loads(completion.choices[0].message.content)
                sec_detail["start_time"] = sec_start_str
                sec_detail["end_time"] = sec_end_str
                results.append({"section_title": sec.get("title"), "details": sec_detail})
            finally:
                if os.path.exists(clip_name): os.remove(clip_name)
        return results

# ================= 任务调度流 =================
def merge_and_slice_data(global_info, asr_data, danmu_data):
    # 保持你原来的组装逻辑
    sections = global_info.get("section_info", [])
    for sec in sections:
        start_sec = time_to_sec(sec.get("start_time"))
        end_sec = time_to_sec(sec.get("end_time"))
        if asr_data:
            sec["section_asr"] = [f"[{i['time_str']}] {i['speaker']}: {i['text']}" for i in asr_data if start_sec <= i['start_sec'] <= end_sec]
        if danmu_data:
            sec["section_danmu"] = [i for i in danmu_data if start_sec <= i.get('timestamp', 0) <= end_sec]
    return global_info

def background_workflow(video_path: str, task_id: str):
    """这是在后台运行的主流程，它会逐步更新 TASK_DB 让前端读取"""
    try:
        p1 = Part1GlobalAnalyzer()
        p2 = Part2ASR()
        p3 = Part3Danmu()
        p4 = Part4SectionDetails()

        with ThreadPoolExecutor(max_workers=3) as top_executor:
            # 1. 并发启动前三个任务
            future_p1 = top_executor.submit(p1.run, video_path, task_id)
            future_p2 = top_executor.submit(p2.run, video_path, task_id)
            future_p3 = top_executor.submit(p3.run, video_path, task_id)

            # 2. 获取 Part 1 结果并立即暴露给前端！
            global_info = future_p1.result() 
            TASK_DB[task_id]["data"]["part1"] = global_info
            
            # 3. 启动 Part 4
            section_info = global_info.get("section_info", [])
            future_p4 = top_executor.submit(p4.run, video_path, section_info, task_id)

            # 4. 等待其余任务并实时更新数据库
            asr_data = future_p2.result()
            TASK_DB[task_id]["data"]["part2"] = asr_data

            danmu_data = future_p3.result()
            TASK_DB[task_id]["data"]["part3"] = danmu_data

            p4_details = future_p4.result()
            TASK_DB[task_id]["data"]["part4"] = p4_details

        # 5. 所有数据均就绪，整合并更新最终状态
        final_sliced_data = merge_and_slice_data(global_info, asr_data, danmu_data)
        TASK_DB[task_id]["data"]["final_result"] = {
            "global_analysis": final_sliced_data,
            "section_details": p4_details
        }
        TASK_DB[task_id]["status"] = "completed"

    except Exception as e:
        TASK_DB[task_id]["status"] = "failed"
        TASK_DB[task_id]["error"] = str(e)
    finally:
        # 清理原视频
        if os.path.exists(video_path):
            os.remove(video_path)

# ================= API 路由 =================

@app.post("/api/analyze")
async def upload_and_start(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """接收视频并分配任务 ID"""
    task_id = uuid.uuid4().hex
    file_location = f"temp_storage/{task_id}_{file.filename}"
    
    # 异步写入文件
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)

    # 初始化任务状态数据库
    TASK_DB[task_id] = {
        "status": "processing",
        "data": {
            "part1": None,  # 全局视频信息
            "part2": None,  # 全局 ASR
            "part3": None,  # 全局弹幕
            "part4": None,  # 章节详情
            "final_result": None
        },
        "error": None
    }

    # 启动后台处理任务，立刻向前端返回 task_id
    background_tasks.add_task(background_workflow, file_location, task_id)
    
    return JSONResponse(content={"task_id": task_id, "message": "任务已在后台启动"})

@app.get("/api/status/{task_id}")
def get_status(task_id: str):
    """前端轮询该接口获取实时进度和局部数据"""
    if task_id not in TASK_DB:
        return JSONResponse(status_code=404, content={"error": "任务不存在或已失效"})
    return TASK_DB[task_id]