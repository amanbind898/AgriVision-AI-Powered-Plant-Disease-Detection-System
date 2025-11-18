import os
import httpx
from typing import Optional

class PerplexityClient:
    """Client for Perplexity API integration"""
    
    def __init__(self):
        self.api_key = os.getenv("PERPLEXITY_API_KEY", "")
        self.base_url = "https://api.perplexity.ai/chat/completions"
        self.model = "sonar"
        
        if self.api_key:
            print(f"✅ Perplexity API key loaded: {self.api_key[:10]}...")
        else:
            print("⚠️  No Perplexity API key found in environment")
    
    async def get_response(
        self,
        message: str,
        context: str = "",
        language: str = "en"
    ) -> str:
        """
        Get response from Perplexity API
        
        Args:
            message: User's question
            context: Additional context (plant/disease info)
            language: Response language (en or hi)
        
        Returns:
            AI response text
        """
        if not self.api_key:
            return self._get_fallback_response(message, context, language)
        
        try:
            # Build system prompt
            system_prompt = self._build_system_prompt(language)
            
            # Build user message with context
            user_message = context + message if context else message
            
            # Make API request
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.base_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message}
                        ],
                        "temperature": 0.7,
                        "max_tokens": 500
                    },
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data["choices"][0]["message"]["content"]
                else:
                    print(f"❌ Perplexity API error: Status {response.status_code}")
                    print(f"Response: {response.text}")
                    return self._get_fallback_response(message, context, language)
        
        except Exception as e:
            print(f"❌ Perplexity API exception: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return self._get_fallback_response(message, context, language)
    
    def _build_system_prompt(self, language: str) -> str:
        """Build system prompt based on language"""
        if language == "hi":
            return """आप एक कृषि विशेषज्ञ हैं जो किसानों को पौधों की बीमारियों, उपचार और कृषि पद्धतियों के बारे में सलाह देते हैं। 
            सरल हिंदी में जवाब दें जो किसान आसानी से समझ सकें। व्यावहारिक और कार्रवाई योग्य सलाह दें।"""
        else:
            return """You are an agricultural expert helping farmers with plant diseases, treatments, and farming practices. 
            Provide clear, practical, and actionable advice. Keep responses concise and easy to understand."""
    
    def _get_fallback_response(self, message: str, context: str, language: str) -> str:
        """Fallback response when API is unavailable"""
        if language == "hi":
            return """मैं अभी Perplexity AI से जुड़ नहीं पा रहा हूं। कृपया बाद में पुनः प्रयास करें या स्थानीय कृषि विशेषज्ञ से परामर्श लें।
            
            सामान्य सुझाव:
            - संक्रमित पत्तियों को हटा दें
            - पौधों के बीच उचित दूरी बनाए रखें
            - अत्यधिक पानी देने से बचें
            - जैविक कवकनाशी का उपयोग करें"""
        else:
            return """I'm currently unable to connect to Perplexity AI. Please try again later or consult a local agricultural expert.
            
            General recommendations:
            - Remove infected leaves
            - Maintain proper plant spacing
            - Avoid overwatering
            - Use organic fungicides when appropriate"""
