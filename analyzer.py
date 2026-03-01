!pip install openai-whisper
!apt update && apt install ffmpeg -y
from google.colab import files
uploaded = files.upload()
import whisper
import os

class MediaAnalyzer:
    def __init__(self):
        print("[*] جاري تحميل نموذج الذكاء الاصطناعي...")
        self.model = whisper.load_model("base")

    def process_media(self, file_path: str):
        if not os.path.exists(file_path):
            return {"error": f"الملف '{file_path}' غير موجود."}

        print(f"[*] بدأ تحليل الملف: {file_path}")
        result = self.model.transcribe(file_path)

        transcript_text = result.get('text', '')
        generated_tags = self._extract_tags(transcript_text)

        return {
            "transcript": transcript_text,
            "tags": generated_tags,
            "confidence_score": 0.95
        }

    def _extract_tags(self, text: str):
        words = text.split()
        tags = list(set([w.strip(",.?!") for w in words if len(w) > 5]))
        return tags[:8]

# تشغيل النظام
file_name = "video.mp4"  # غيّري الاسم إذا ملفك مختلف

ai_service = MediaAnalyzer()
results = ai_service.process_media(file_name)

if "error" in results:
    print(f"[-] {results['error']}")
else:
    print("\n" + "="*40)
    print("النص المستخرج:\n")
    print(results["transcript"])
    print("\nالوسوم الذكية:")
    print(results["tags"])
    print("="*40)
