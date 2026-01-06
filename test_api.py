"""
Test Gemini API with new key and gemini-2.5-flash model
"""
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

API_KEY = os.getenv('GEMINI_API_KEY')

print("=" * 50)
print("🔑 Testing Gemini API Key")
print("=" * 50)
print(f"\nAPI Key: {API_KEY[:10]}...{API_KEY[-5:]}" if API_KEY else "API Key not found!")

if not API_KEY:
    print("❌ No API key found in .env file!")
    exit(1)

try:
    genai.configure(api_key=API_KEY)
    
    # Test gemini-2.5-flash
    print("\n🧪 Testing gemini-2.5-flash...")
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content("Say 'Hello World' in one line")
    
    if response and response.text:
        print(f"✅ SUCCESS! Model is working!")
        print(f"📝 Response: {response.text.strip()}")
        print("\n🎉 Your API is ready to use!")
    else:
        print("❌ No response received")

except Exception as e:
    error_msg = str(e)
    print(f"\n❌ Error: {error_msg}")
    
    if "429" in error_msg or "quota" in error_msg.lower():
        print("\n⚠️ Quota exceeded! Wait a few minutes or check your billing.")
    elif "API_KEY_INVALID" in error_msg or "expired" in error_msg.lower():
        print("\n⚠️ API key is invalid or expired!")
        print("   Get a new key from: https://makersuite.google.com/app/apikey")
    elif "not found" in error_msg.lower():
        print("\n⚠️ Model 'gemini-2.5-flash' not found!")
        print("   Trying alternative models...")
        
        # Try alternative models
        alt_models = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']
        for alt in alt_models:
            try:
                print(f"\n   Testing {alt}...")
                model = genai.GenerativeModel(alt)
                response = model.generate_content("Say hello")
                if response.text:
                    print(f"   ✅ {alt} works! Use this model instead.")
                    break
            except:
                print(f"   ❌ {alt} failed")

print("\n" + "=" * 50)
