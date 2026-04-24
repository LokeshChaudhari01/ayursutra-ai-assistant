from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from typing import List, Optional, Dict
import json
import re
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

app = FastAPI(title="Ayursutra AI Assistant", version="1.0.0")

# CORS middleware for Next.js integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Removed static files mounting - focusing on API only

# Initialize Google Gemini
google_api_key = os.getenv("GOOGLE_API_KEY")
model = None
if google_api_key:
    try:
        # Configure with API key
        genai.configure(api_key=google_api_key)
        
        # Try different model names - using actual available models from your API key
        model_configs = [
            # Latest 2.5 series - most capable
            {'name': 'models/gemini-2.5-flash', 'is_beta': False},
            {'name': 'models/gemini-2.5-pro', 'is_beta': False},
            # Latest stable pointers
            {'name': 'models/gemini-flash-latest', 'is_beta': False},
            {'name': 'models/gemini-pro-latest', 'is_beta': False},
            # 2.0 series fallback
            {'name': 'models/gemini-2.0-flash', 'is_beta': False},
            {'name': 'models/gemini-2.0-flash-001', 'is_beta': False}
        ]
        
        for config in model_configs:
            model_name = config['name']
            is_beta = config['is_beta']
            
            try:
                print(f"🔄 Trying {model_name} ({'beta' if is_beta else 'stable'} API)...")
                
                if is_beta:
                    # For beta models, we need to ensure proper configuration
                    # The Python client should automatically handle v1beta for 1.5 models
                    model = genai.GenerativeModel(model_name)
                else:
                    # For stable models
                    model = genai.GenerativeModel(model_name)
                
                # Test the model with a simple generation
                test_response = model.generate_content("Hello")
                print(f"✅ Successfully initialized: {model_name}")
                break
                
            except Exception as e:
                print(f"⚠️ Failed to initialize {model_name}: {str(e)[:100]}")
                continue
        
        if not model:
            print("❌ Could not initialize any Gemini model")
            
    except Exception as e:
        print(f"⚠️ Could not configure Gemini: {e}")
        model = None

# Pydantic models
class ChatMessage(BaseModel):
    message: str
    search_medicines: Optional[bool] = False  # Option to search medicines in chat
    user_id: Optional[str] = "anonymous"

# Removed unused models - focusing on chat only

class ChatResponse(BaseModel):
    response: str
    medicine_results: Optional[List[Dict]] = None
    confidence: Optional[float] = None

# Ayurvedic knowledge base for better responses
AYURVEDIC_KNOWLEDGE = """
You are an expert Ayurvedic AI assistant. You have deep knowledge of:
- Ayurvedic principles (Vata, Pitta, Kapha doshas)
- Traditional herbs and medicines
- Home remedies and natural treatments
- Ayurvedic lifestyle recommendations
- Seasonal health practices

Always provide helpful, accurate information while reminding users to consult qualified practitioners for serious conditions.
"""

# Web scraping configuration
PHARMACY_SOURCES = {
    "1mg": {
        "url": "https://www.1mg.com/search/all?name={query}",
        "selectors": {
            "container": ["div[data-testid='product-card']", ".product-card", ".medicine-unit-wrap"],
            "name": ["h3", "h4", ".product-name", "[data-testid='product-name']"],
            "price": [".price", ".cost", "[data-testid='price']", ".price-box span"],
            "rating": [".rating", ".stars", "[data-testid='rating']"]
        }
    },
    "netmeds": {
        "url": "https://www.netmeds.com/catalogsearch/result?q={query}",
        "selectors": {
            "container": [".product-item", ".medicine-card", ".product-wrapper"],
            "name": [".product-name", "h3", "h4"],
            "price": [".price", ".final-price", ".offer-price"],
            "rating": [".rating", ".star-rating"]
        }
    },
    "pharmeasy": {
        "url": "https://pharmeasy.in/search/all?name={query}",
        "selectors": {
            "container": [".ProductCard_medicineUnitWrapper", ".product-card"],
            "name": [".ProductCard_medicineName", ".product-name"],
            "price": [".ProductCard_gcdDiscountContainer", ".price"],
            "rating": [".ProductCard_ratingWrapper", ".rating"]
        }
    },
    "baidyanath": {
        "url": "https://www.baidyanath.com/search?q={query}",
        "selectors": {
            "container": [".product-item", ".product-card", ".grid-product"],
            "name": [".product-title", "h3", "h4", ".product-name"],
            "price": [".price", ".money", ".product-price", ".current-price"],
            "rating": [".reviews", ".rating", ".stars"]
        }
    },
    "dabur": {
        "url": "https://www.dabur.com/search?q={query}",
        "selectors": {
            "container": [".product-item", ".product-card", ".search-result"],
            "name": [".product-title", ".product-name", "h3", "h4"],
            "price": [".price", ".product-price", ".cost"],
            "rating": [".rating", ".reviews", ".stars"]
        }
    },
    "patanjali": {
        "url": "https://www.patanjaliayurveda.net/search?q={query}",
        "selectors": {
            "container": [".product-item", ".product-card", ".grid-item"],
            "name": [".product-title", ".product-name", "h3"],
            "price": [".price", ".product-price", ".money"],
            "rating": [".rating", ".reviews"]
        }
    },
    "apollopharmacy": {
        "url": "https://www.apollopharmacy.in/search-medicines/{query}",
        "selectors": {
            "container": [".ProductCard", ".product-card", ".medicine-card"],
            "name": [".ProductName", ".product-name", "h3"],
            "price": [".Price", ".price", ".cost"],
            "rating": [".Rating", ".rating", ".stars"]
        }
    },
    "zandu": {
        "url": "https://www.zandu.in/search?q={query}",
        "selectors": {
            "container": [".product-item", ".product-card"],
            "name": [".product-title", ".product-name"],
            "price": [".price", ".product-price"],
            "rating": [".rating", ".reviews"]
        }
    },
    "himalaya": {
        "url": "https://himalayawellness.in/search?q={query}",
        "selectors": {
            "container": [".product-item", ".product-card"],
            "name": [".product-title", ".product-name"],
            "price": [".price", ".product-price"],
            "rating": [".rating", ".reviews"]
        }
    }
}

async def scrape_medicine_prices(medicine_name: str, max_results: int = 5) -> List[Dict]:
    """Scrape medicine prices from multiple sources with dynamic selectors"""
    results = []
    query = medicine_name.replace(' ', '%20')
    
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=10),
        headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    ) as session:
        
        for source_name, config in list(PHARMACY_SOURCES.items())[:4]:  # Limit to 4 sources for speed
            try:
                url = config["url"].format(query=query)
                
                async with session.get(url, timeout=8) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Try different container selectors
                        medicines = []
                        for selector in config["selectors"]["container"]:
                            medicines = soup.select(selector)[:3]
                            if medicines:
                                break
                        
                        for med in medicines:
                            try:
                                # Extract name
                                name = None
                                for name_selector in config["selectors"]["name"]:
                                    name_elem = med.select_one(name_selector)
                                    if name_elem:
                                        name = name_elem.get_text(strip=True)
                                        break
                                
                                # Extract price
                                price = None
                                for price_selector in config["selectors"]["price"]:
                                    price_elem = med.select_one(price_selector)
                                    if price_elem:
                                        price = price_elem.get_text(strip=True)
                                        break
                                
                                # Extract rating if available
                                rating = "N/A"
                                for rating_selector in config["selectors"]["rating"]:
                                    rating_elem = med.select_one(rating_selector)
                                    if rating_elem:
                                        rating = rating_elem.get_text(strip=True)
                                        break
                                
                                if name and price:
                                    results.append({
                                        "Medicine": name[:100],  # Limit length
                                        "Price": price,
                                        "Source": source_name.title(),
                                        "Stock": "Check Availability",
                                        "Rating": rating,
                                        "Link": url
                                    })
                                    
                            except Exception as item_error:
                                print(f"Error extracting item from {source_name}: {item_error}")
                                continue
                                
            except Exception as source_error:
                print(f"Error scraping {source_name}: {source_error}")
                continue
    
    return results[:max_results]

# LLM Integration
async def get_ai_response(message: str, search_medicines: bool = False) -> ChatResponse:
    """Get AI response using Google Gemini or fallback to rule-based system"""
    print(f"🤖 Processing message: {message[:50]}...")
    
    try:
        if google_api_key and model:
            print("🚀 Using Google Gemini...")

            # Provide a comprehensive prompt and let the LLM handle context
            full_prompt = f"""
You are Ayursutra, an expert Ayurvedic health and wellness assistant. 

User's message: {message}

Instructions:
1. If the user is asking a health, wellness, or Ayurveda-related question:
   - Understand and acknowledge their concern empathetically.
   - Suggest relevant Ayurvedic remedies (herbs, medicines, dosha balancing if applicable).
   - Provide practical home remedies, dietary, and lifestyle suggestions.
   - Provide detailed, helpful answers. Use bullet points and clear formatting for readability.
   - Remind them to consult a qualified Ayurvedic doctor for serious conditions.
2. If the user asks general or unrelated questions, gently explain that your expertise is in Ayurveda and health, and answer as best as you can while guiding them back to wellness topics.
3. If the user greets you, respond warmly and ask how you can help with their health today.

Keep your tone warm, respectful, and professional. Do not give vague or evasive answers if the user has a genuine question.
"""

            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: model.generate_content(full_prompt)
            )

            ai_response = response.text.strip()
            return ChatResponse(
                response=ai_response,
                medicine_results=None,
                confidence=0.9
            )

        else:
            print("⚠️ No API key or model, using fallback...")
            return get_rule_based_response(message)

    except Exception as e:
        print(f"❌ AI response error: {e}")
        return get_rule_based_response(message)


def extract_medicine_suggestions(text: str) -> List[str]:
    """Extract medicine suggestions from AI response"""
    # Extended list of common Ayurvedic medicines and herbs
    ayurvedic_medicines = [
        'Ashwagandha', 'Triphala', 'Tulsi', 'Giloy', 'Guduchi', 'Turmeric', 'Ginger', 'Neem', 
        'Brahmi', 'Shankhpushpi', 'Amla', 'Arjuna', 'Bala', 'Bhringraj', 'Chyavanprash',
        'Dashmool', 'Fenugreek', 'Garlic', 'Haritaki', 'Jatamansi', 'Kumari', 'Licorice',
        'Manjistha', 'Moringa', 'Nutmeg', 'Pippali', 'Punarnava', 'Rasayana', 'Shatavari',
        'Tagar', 'Vidanga', 'Yashtimadhu', 'Zinc', 'Aloevera', 'Cardamom', 'Cinnamon',
        'Sitopaladi Churna', 'Taleesadi Churna', 'Avipattikar Churna', 'Hingvastak Churna',
        'Godanti Mishran', 'Shirashooladi Vajra Ras', 'Panchakola Churna', 'Trikatu Churna',
        'Madana Phala', 'Saindhava Lavana', 'Honey', 'Ghee', 'Rock Salt'
    ]
    
    # Create pattern from the medicine list
    pattern = r'\b(?:' + '|'.join(re.escape(med) for med in ayurvedic_medicines) + r')\b'
    medicines = re.findall(pattern, text, re.IGNORECASE)
    return list(set(medicines))

def get_rule_based_response(message: str) -> ChatResponse:
    """Fallback rule-based response system"""
    message_lower = message.lower()
    
    # Check if this is a health-related query
    health_keywords = ['pain', 'ache', 'sick', 'disease', 'cure', 'medicine', 'remedy', 'treatment', 'symptom', 'health', 'ayurveda', 'herbal', 'dosage', 'tablet', 'capsule', 'syrup', 'oil', 'powder', 'churna', 'vati', 'ras', 'asava', 'aristha', 'cold', 'cough', 'fever', 'throat', 'headache', 'migraine', 'digestion', 'stomach', 'acidity', 'gas', 'bloating', 'stress', 'anxiety', 'sleep', 'insomnia', 'immunity', 'weakness', 'fatigue']
    is_health_query = any(keyword in message_lower for keyword in health_keywords)
    
    if not is_health_query:
        # General conversation - no medical advice
        return ChatResponse(
            response="🙏 Hello! I'm your Ayurvedic wellness assistant.\n\nI'm here to help when you have specific health concerns or questions about Ayurvedic remedies. Feel free to ask me about:\n\n• Symptoms you're experiencing\n• Natural remedies for health issues\n• Ayurvedic lifestyle advice\n• Traditional medicine recommendations\n\nHow can I assist you with your health and wellness today? 🌿",
            suggested_medicines=None,
            confidence=0.6
        )
    
    # Health-related responses with medicine recommendations
    if any(word in message_lower for word in ['cold', 'cough', 'fever', 'throat']):
        return ChatResponse(
            response="For cold and cough, try these Ayurvedic remedies:\n\n🌿 **Home Remedies:**\n• Warm water with honey and ginger\n• Tulsi tea 3 times daily\n• Turmeric milk before bed\n• Steam inhalation with eucalyptus\n\n💊 **Recommended Medicines:**\n• Sitopaladi Churna - 1 tsp with honey\n• Taleesadi Churna - for dry cough\n\n⚠️ Consult an Ayurvedic doctor if symptoms persist.",
            suggested_medicines=["Sitopaladi Churna", "Taleesadi Churna", "Tulsi", "Honey"],
            confidence=0.8
        )
    
    elif any(word in message_lower for word in ['headache', 'migraine', 'head pain']):
        return ChatResponse(
            response="For headaches, try these Ayurvedic approaches:\n\n🌿 **Immediate Relief:**\n• Apply peppermint oil to temples\n• Drink ginger tea with lemon\n• Practice deep breathing (Pranayama)\n• Rest in a dark, quiet room\n\n💊 **Ayurvedic Medicines:**\n• Godanti Mishran - for chronic headaches\n• Shirashooladi Vajra Ras - for severe pain\n\n⚠️ If headaches are frequent, consult a doctor.",
            suggested_medicines=["Godanti Mishran", "Shirashooladi Vajra Ras", "Ginger"],
            confidence=0.8
        )
    
    elif any(word in message_lower for word in ['digestion', 'stomach', 'acidity', 'gas', 'bloating']):
        return ChatResponse(
            response="For digestive issues, here's what Ayurveda recommends:\n\n🌿 **Dietary Changes:**\n• Drink warm water with lemon in morning\n• Chew fennel seeds after meals\n• Avoid cold drinks with food\n• Eat meals at regular times\n\n💊 **Ayurvedic Medicines:**\n• Triphala - before bed for overall digestion\n• Avipattikar Churna - for acidity\n• Hingvastak Churna - for gas and bloating\n\n⚠️ Maintain regular eating schedule.",
            suggested_medicines=["Avipattikar Churna", "Hingvastak Churna", "Triphala"],
            confidence=0.8
        )
    
    elif any(word in message_lower for word in ['stress', 'anxiety', 'sleep', 'insomnia', 'tension']):
        return ChatResponse(
            response="For stress and sleep issues, Ayurveda suggests:\n\n🌿 **Lifestyle Changes:**\n• Practice meditation daily\n• Warm oil massage before bed\n• Avoid screens 1 hour before sleep\n• Drink chamomile or brahmi tea\n\n💊 **Ayurvedic Medicines:**\n• Ashwagandha - for stress relief\n• Brahmi - for mental clarity\n• Jatamansi - for better sleep\n\n⚠️ Maintain consistent sleep schedule.",
            suggested_medicines=["Ashwagandha", "Brahmi", "Jatamansi"],
            confidence=0.8
        )
    
    elif any(word in message_lower for word in ['immunity', 'immune', 'weakness', 'energy', 'fatigue']):
        return ChatResponse(
            response="To boost immunity and energy naturally:\n\n🌿 **Daily Routine:**\n• Start day with warm water and honey\n• Include ginger, turmeric in diet\n• Get adequate sunlight\n• Practice yoga or light exercise\n\n💊 **Immunity Boosters:**\n• Chyavanprash - 1 tsp daily\n• Giloy tablets - natural immunity booster\n• Amla juice - high in Vitamin C\n\n⚠️ Maintain balanced diet and regular exercise.",
            suggested_medicines=["Chyavanprash", "Giloy", "Amla", "Ashwagandha"],
            confidence=0.8
        )
    
    else:
        # Health query but no specific symptom matched
        return ChatResponse(
            response="🙏 I understand you have a health concern. Could you please provide more specific details about your symptoms?\n\nI can help with:\n• **Common symptoms** - cold, headache, digestion issues, stress\n• **Specific medicines** - traditional Ayurvedic remedies\n• **Home remedies** - natural healing solutions\n• **Lifestyle advice** - based on Ayurvedic principles\n\nThe more details you share about your specific condition, the better I can assist you with personalized Ayurvedic guidance! 🌿",
            suggested_medicines=["Triphala", "Ashwagandha", "Tulsi"],
            confidence=0.6
        )
# API Endpoints
@app.get("/")
async def root():
    return FileResponse('static/index.html')

@app.get("/api")
async def api_info():
    return {
        "message": "Welcome to Ayursutra AI Assistant",
        "version": "1.0.0",
        "status": "running",
        "ai_enabled": bool(google_api_key and model),
        "model_status": "Initialized" if model else "Fallback mode",
        "endpoints": ["/chat", "/medicines/search", "/medicines/list", "/symptoms/analyze", "/medicines/categories"]
    }

@app.get("/debug/models")
async def list_models():
    """Debug endpoint to list available Gemini models"""
    if not google_api_key:
        return {"error": "No API key configured"}
    
    try:
        models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                models.append({
                    "name": m.name,
                    "display_name": getattr(m, 'display_name', 'No display name'),
                    "description": getattr(m, 'description', 'No description'),
                    "supported_methods": m.supported_generation_methods
                })
        return {
            "available_models": models,
            "total_count": len(models),
            "current_model": model.model_name if model else "None"
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/debug/test-gemini")
async def test_gemini():
    """Test Gemini model directly"""
    if not google_api_key:
        return {"error": "No API key configured"}
    
    if not model:
        return {"error": "No model initialized"}
    
    try:
        # Use optimized config for faster testing
        generation_config = genai.types.GenerationConfig(
            temperature=0.3,
            max_output_tokens=50,
            candidate_count=1
        )
        
        response = model.generate_content("Say hello in one word", generation_config=generation_config)
        return {
            "success": True,
            "model_name": model.model_name,
            "response": response.text,
            "usage": getattr(response, 'usage_metadata', 'No usage data')
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/debug/api-key")
async def validate_api_key():
    """Validate API key and setup"""
    if not google_api_key:
        return {"error": "No GOOGLE_API_KEY found in environment"}
    
    # Check API key format
    if not google_api_key.startswith('AIzaSy'):
        return {"error": "Invalid API key format. Should start with 'AIzaSy'"}
    
    try:
        # Test basic API connection
        models_list = list(genai.list_models())
        return {
            "api_key_valid": True,
            "api_key_prefix": google_api_key[:10] + "...",
            "total_models_found": len(models_list),
            "api_connection": "Success"
        }
    except Exception as e:
        return {
            "api_key_valid": False,
            "error": str(e),
            "suggestion": "Check if API key is correct and has proper permissions"
        }

@app.get("/debug/list-all-models")
async def list_all_available_models():
    """List all available models with your API key"""
    if not google_api_key:
        return {"error": "No GOOGLE_API_KEY found"}
    
    try:
        # Get all available models
        models_list = list(genai.list_models())
        
        model_details = []
        for model in models_list:
            model_info = {
                "name": model.name,
                "base_model_id": getattr(model, 'base_model_id', 'N/A'),
                "version": getattr(model, 'version', 'N/A'),
                "display_name": getattr(model, 'display_name', 'N/A'),
                "description": getattr(model, 'description', 'N/A'),
                "input_token_limit": getattr(model, 'input_token_limit', 'N/A'),
                "output_token_limit": getattr(model, 'output_token_limit', 'N/A'),
                "supported_generation_methods": getattr(model, 'supported_generation_methods', [])
            }
            model_details.append(model_info)
        
        # Also get just the model names for easy reference
        model_names = [model.name for model in models_list]
        
        return {
            "success": True,
            "total_models": len(models_list),
            "model_names": model_names,
            "detailed_models": model_details,
            "suggestion": "Use one of these exact model names in your code"
        }
        
    except Exception as e:
        return {
            "error": f"Failed to list models: {str(e)}",
            "suggestion": "Check if your API key has the correct permissions"
        }

@app.post("/chat", response_model=ChatResponse)
async def chat_with_ai(chat_message: ChatMessage):
    """Streamlined chat with integrated medicine search"""
    try:
        response = await get_ai_response(
            message=chat_message.message,
            search_medicines=chat_message.search_medicines
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

# Removed separate medicine search endpoint - integrated into chat

# Removed medicine list endpoint - streamlined to chat only

# Removed symptom analysis endpoint - integrated into main chat

@app.get("/medicines/categories")
async def get_medicine_categories():
    """Get medicine categories and their medicines"""
    categories_with_medicines = {}
    for category, medicines in AYURVEDIC_MEDICINE_CATEGORIES.items():
        categories_with_medicines[category] = {
            "name": category.replace("_", " ").title(),
            "medicines": medicines,
            "count": len(medicines)
        }
    
    return {
        "categories": categories_with_medicines,
        "total_categories": len(AYURVEDIC_MEDICINE_CATEGORIES),
        "description": "Ayurvedic medicine categories based on traditional usage"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for Railway"""
    return {
        "status": "healthy", 
        "message": "Ayursutra AI Assistant is running",
        "features": [
            "Google Gemini AI Integration",
            "Real-time Medicine Search",
            "Dynamic Knowledge Base",
            "Symptom Analysis",
            "Category-based Recommendations"
        ]
    }

# Dynamic Medicine Categories Database
AYURVEDIC_MEDICINE_CATEGORIES = {
    "digestive": ["Triphala", "Avipattikar Churna", "Hingvastak Churna", "Ajwain", "Hing"],
    "respiratory": ["Sitopaladi Churna", "Taleesadi Churna", "Tulsi", "Honey", "Ginger"],
    "immunity": ["Giloy", "Guduchi", "Ashwagandha", "Chyavanprash", "Amla"],
    "stress_anxiety": ["Brahmi", "Shankhpushpi", "Jatamansi", "Ashwagandha", "Tagar"],
    "pain_inflammation": ["Turmeric", "Ginger", "Dashmool", "Nirgundi", "Mahanarayana Oil"],
    "skin_hair": ["Neem", "Manjistha", "Bhringraj", "Aloevera", "Kumkumadi Oil"],
    "women_health": ["Shatavari", "Lodhra", "Pushyanug Churna", "Kumaryasava", "Dashmool"],
    "men_health": ["Ashwagandha", "Safed Musli", "Kapikachhu", "Gokshura", "Shilajit"],
    "heart_circulation": ["Arjuna", "Garlic", "Punarnava", "Hridayarnava Ras", "Terminalia"],
    "diabetes": ["Karela", "Jamun", "Methi", "Vijaysar", "Gudmar"],
    "liver_detox": ["Bhumi Amla", "Kalmegh", "Kutki", "Punarnava", "Liv-52"],
    "joints_bones": ["Guggul", "Shallaki", "Rasna", "Nirgundi", "Yograj Guggul"]
}

def get_medicine_suggestions_by_category(symptoms: str) -> List[str]:
    """Get medicine suggestions based on symptom categories"""
    suggestions = []
    symptoms_lower = symptoms.lower()
    
    # Map symptoms to categories
    symptom_mapping = {
        "digestive": ["stomach", "acidity", "indigestion", "gas", "bloating", "constipation"],
        "respiratory": ["cough", "cold", "asthma", "breathing", "chest", "throat"],
        "immunity": ["fever", "infection", "weak", "immunity", "frequent illness"],
        "stress_anxiety": ["stress", "anxiety", "depression", "sleep", "mental", "worry"],
        "pain_inflammation": ["pain", "inflammation", "swelling", "arthritis", "headache"],
        "skin_hair": ["skin", "hair", "acne", "rash", "eczema", "dandruff"],
        "women_health": ["menstrual", "periods", "pregnancy", "fertility", "hormonal"],
        "men_health": ["stamina", "energy", "vitality", "testosterone", "strength"],
        "heart_circulation": ["heart", "blood pressure", "circulation", "cholesterol"],
        "diabetes": ["sugar", "diabetes", "blood sugar", "glucose"],
        "liver_detox": ["liver", "detox", "cleanse", "toxins", "jaundice"],
        "joints_bones": ["joint", "bone", "arthritis", "stiffness", "mobility"]
    }
    
    # Find matching categories
    for category, keywords in symptom_mapping.items():
        if any(keyword in symptoms_lower for keyword in keywords):
            suggestions.extend(AYURVEDIC_MEDICINE_CATEGORIES.get(category, []))
    
    return list(set(suggestions))[:10]  # Return unique suggestions, limit to 10

async def get_dynamic_medicine_data(medicine_name: str) -> List[Dict]:
    """Get dynamic medicine data by combining web scraping with knowledge base"""
    
    # Get real-time data from web scraping
    realtime_data = await scrape_medicine_prices(medicine_name, max_results=5)
    
    # Add knowledge base information
    knowledge_data = []
    
    # Check if medicine exists in our categories
    for category, medicines in AYURVEDIC_MEDICINE_CATEGORIES.items():
        if any(med.lower() in medicine_name.lower() for med in medicines):
            knowledge_data.append({
                "Medicine": medicine_name,
                "Category": category.replace("_", " ").title(),
                "Type": "Ayurvedic Medicine",
                "Usage": f"Traditional use for {category.replace('_', ' ')} related conditions",
                "Source": "Knowledge Base",
                "Note": "Consult an Ayurvedic practitioner for proper dosage"
            })
    
    return realtime_data + knowledge_data

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    
    # Use localhost for local development, 0.0.0.0 for Railway
    host = "127.0.0.1" if not os.environ.get("RAILWAY_ENVIRONMENT") else "0.0.0.0"
    
    print(f"🚀 Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)

