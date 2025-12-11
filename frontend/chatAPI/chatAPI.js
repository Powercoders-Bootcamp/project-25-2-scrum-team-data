
        const API_URL = 'http://127.0.0.1:8000';


        async function checkBackendHealth() {
            try {
                const response = await fetch(`${API_URL}/health`);
                if (response.ok) {
                    const data = await response.json();
                    console.log('Backend health check:', data);
                    return true;
                }
                return false;
            } catch (error) {
                console.error('Backend health check failed:', error);
                return false;
            }
        }

        export async function sendMessageToChat () {

        let user_id, session_id, messages_history;
        
        try {
            user_id = JSON.parse(localStorage.getItem('user_id'));
        } catch (e) {
            console.error('Error parsing user_id:', e);
            user_id = null;
        }
        
        try {
            session_id = JSON.parse(sessionStorage.getItem('session_id'));
        } catch (e) {
            console.error('Error parsing session_id:', e);
            session_id = null;
        }
        
        try {
            messages_history = JSON.parse(sessionStorage.getItem('messages-history')) || [];
        } catch (e) {
            console.error('Error parsing messages_history:', e);
            messages_history = [];
        }
        
        console.log('Sending request with:', { session_id, user_id, messages_count: messages_history.length });
        
        // session_id veya user_id yoksa hata fırlat
        if (!session_id) {
            console.error('Session ID is missing');
            throw new Error('Session ID is missing. Please refresh the page.');
        }
        if (!user_id) {
            console.error('User ID is missing');
            throw new Error('User ID is missing. Please refresh the page.');
        }
        
        try {
            console.log(`Attempting to fetch from ${API_URL}/api/chat`);
            const response = await fetch(`${API_URL}/api/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    session_id,
                    user_id,
                    messages: messages_history,
                    top_k: 5,
                    use_reranker: false
                }),
            });

            console.log('Response status:', response.status, response.statusText);

            // Check HTTP status
            if (!response.ok) {
                let errorData;
                try {
                    errorData = await response.json();
                } catch (e) {
                    const text = await response.text();
                    errorData = { detail: text || 'Unknown error' };
                }
                console.error('API Error:', errorData);
                // Handle both string and object detail formats
                const errorMessage = typeof errorData.detail === 'string' 
                    ? errorData.detail
                    : (errorData.detail?.message || errorData.message || JSON.stringify(errorData.detail) || 'Unknown error');
                throw new Error(`API Error: ${response.status} - ${errorMessage}`);
            }

            // Parse response
            const responseData = await response.json();
            console.log('API Response:', responseData);
            
            if (!responseData.answer) {
                console.error('No answer in response:', responseData);
                throw new Error('No answer in response');
            }

            return responseData.answer;
        } catch (error) {
            console.error('sendMessageToChat error:', error);
            
            // "Failed to fetch" hatası genellikle backend'e bağlanılamadığında oluşur
            if (error.message === 'Failed to fetch' || error.name === 'TypeError' || error.message.includes('fetch')) {
                // Backend'in çalışıp çalışmadığını kontrol et
                const isBackendHealthy = await checkBackendHealth();
                if (!isBackendHealthy) {
                    throw new Error('Backend server\'a bağlanılamıyor. Lütfen backend\'in çalıştığından emin olun (http://127.0.0.1:8000). Backend\'i başlatmak için: cd project-25-2-scrum-team-data && python -m uvicorn backend.main:app --reload');
                } else {
                    throw new Error('Backend çalışıyor ancak API çağrısı başarısız oldu. Lütfen console loglarını kontrol edin.');
                }
            }
            
            throw error;
        }
    }


