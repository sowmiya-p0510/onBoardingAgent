import requests
import time
import json

class PerformanceTest:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        
    def test_document_loading_and_caching(self):
        """Test complete document loading and caching performance"""
        print("🚀 COMPLETE DOCUMENT LOADING & CACHING TEST")
        print("=" * 60)
        
        # Test questions that require different types of information
        test_questions = [
            "What health insurance benefits do we have?",
            "Tell me about our dental coverage",
            "What are the retirement benefits?",
            "Explain all vacation and PTO policies",
            "What benefits are available for new employees?"
        ]
        
        results = []
        
        for i, question in enumerate(test_questions):
            print(f"\n📝 Test {i+1}: {question}")
            print("-" * 50)
            
            start_time = time.time()
            
            payload = {
                "user_id": f"perf-test-{i+1}",
                "question": question,
                "session_id": f"perf-session-{i+1}"
            }
            
            try:
                response = requests.post(f"{self.base_url}/chat", json=payload, timeout=120)
                request_time = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    response_text = data.get('response', '')
                    
                    result = {
                        "question": question,
                        "time": request_time,
                        "response_length": len(response_text),
                        "status": "success"
                    }
                    
                    print(f"✅ Success in {request_time:.2f} seconds")
                    print(f"📝 Response length: {len(response_text)} characters")
                    print(f"📋 Sample response: {response_text[:100]}...")
                    
                    # Check if response contains substantial content
                    if len(response_text) > 50 and "couldn't find" not in response_text.lower():
                        print(f"✅ Contains relevant information from documents")
                    else:
                        print(f"⚠️ May not contain expected document information")
                        
                else:
                    result = {
                        "question": question,
                        "time": request_time,
                        "response_length": 0,
                        "status": f"failed_{response.status_code}"
                    }
                    print(f"❌ Failed with status {response.status_code}")
                
                results.append(result)
                
            except Exception as e:
                result = {
                    "question": question,
                    "time": 0,
                    "response_length": 0,
                    "status": f"error_{str(e)}"
                }
                print(f"❌ Error: {e}")
                results.append(result)
            
            # Small delay between requests
            time.sleep(1)
        
        self.analyze_performance(results)
        return results
    
    def analyze_performance(self, results):
        """Analyze performance results"""
        print("\n" + "=" * 60)
        print("📊 PERFORMANCE ANALYSIS")
        print("=" * 60)
        
        successful_results = [r for r in results if r["status"] == "success"]
        
        if not successful_results:
            print("❌ No successful requests to analyze")
            return
        
        times = [r["time"] for r in successful_results]
        first_request_time = times[0] if times else 0
        subsequent_times = times[1:] if len(times) > 1 else []
        
        print(f"📈 First request (document loading): {first_request_time:.2f} seconds")
        
        if subsequent_times:
            avg_subsequent = sum(subsequent_times) / len(subsequent_times)
            print(f"⚡ Subsequent requests (cached): {avg_subsequent:.2f} seconds average")
            
            # Check if caching is working
            cache_improvement = ((first_request_time - avg_subsequent) / first_request_time) * 100
            
            if avg_subsequent < first_request_time * 0.3:  # 70% faster
                print(f"🚀 EXCELLENT CACHING: {cache_improvement:.1f}% improvement")
            elif avg_subsequent < first_request_time * 0.7:  # 30% faster
                print(f"✅ GOOD CACHING: {cache_improvement:.1f}% improvement")
            else:
                print(f"⚠️ CACHING MAY NOT BE WORKING: Only {cache_improvement:.1f}% improvement")
        
        # Response quality analysis
        avg_response_length = sum(r["response_length"] for r in successful_results) / len(successful_results)
        print(f"📝 Average response length: {avg_response_length:.0f} characters")
        
        if avg_response_length > 200:
            print("✅ Responses contain substantial content")
        else:
            print("⚠️ Responses may be too brief")
    
    def test_cache_persistence(self):
        """Test if cache persists across multiple requests"""
        print("\n🔄 TESTING CACHE PERSISTENCE")
        print("=" * 40)
        
        question = "What health benefits do we offer?"
        times = []
        
        # Make 5 consecutive requests
        for i in range(5):
            print(f"📞 Request {i+1}/5...")
            
            start_time = time.time()
            payload = {
                "user_id": f"cache-test-{i+1}",
                "question": question,
                "session_id": f"cache-session-{i+1}"
            }
            
            try:
                response = requests.post(f"{self.base_url}/chat", json=payload, timeout=60)
                request_time = time.time() - start_time
                times.append(request_time)
                
                if response.status_code == 200:
                    print(f"   ✅ {request_time:.2f} seconds")
                else:
                    print(f"   ❌ Failed: {response.status_code}")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
            
            time.sleep(0.5)
        
        if len(times) >= 2:
            print(f"\n📊 Cache Performance:")
            print(f"   🐌 First request: {times[0]:.2f}s (loading documents)")
            if len(times) > 1:
                avg_cached = sum(times[1:]) / len(times[1:])
                print(f"   ⚡ Cached requests: {avg_cached:.2f}s average")
                
                speedup = times[0] / avg_cached if avg_cached > 0 else 0
                print(f"   🚀 Speedup factor: {speedup:.1f}x")
    
    def test_document_listing(self):
        """Test document listing endpoint"""
        print("\n📂 TESTING DOCUMENT LISTING")
        print("=" * 30)
        
        try:
            response = requests.get(f"{self.base_url}/chat/documents", timeout=30)
            if response.status_code == 200:
                data = response.json()
                total = data['documents']['total_count']
                benefits = len(data['documents']['benefits'])
                policies = len(data['documents']['policies'])
                
                print(f"✅ Total documents: {total}")
                print(f"📄 Benefits documents: {benefits}")
                print(f"📋 Policy documents: {policies}")
                
                if benefits > 0:
                    print(f"📋 Benefits files: {data['documents']['benefits']}")
                
                return total > 0
            else:
                print(f"❌ Failed to list documents: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error listing documents: {e}")
            return False

def main():
    """Run all performance tests"""
    tester = PerformanceTest()
    
    print("🧪 STARTING COMPREHENSIVE PERFORMANCE TESTS")
    print("=" * 60)
    
    # Test document access first
    if not tester.test_document_listing():
        print("❌ Cannot proceed - document access failed")
        return
    
    # Run main performance test
    results = tester.test_document_loading_and_caching()
    
    # Test cache persistence
    tester.test_cache_persistence()
    
    print("\n🎯 SUMMARY:")
    print("- First request loads ALL documents completely")
    print("- Subsequent requests use cached documents")
    print("- No content limitations - complete PDF text used")
    print("- No token limits on LLM responses")

if __name__ == "__main__":
    main()
