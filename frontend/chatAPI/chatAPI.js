
        const API_URL = 'http://127.0.0.1:8000';

        
        const user_id = JSON.parse(localStorage.getItem('user_id'));
        const session_id = JSON.parse(sessionStorage.getItem('session_id'));


        export async function sendMessageToChat () {
        const messages_history = JSON.parse(sessionStorage.getItem('messages-history'));
                const data  = await fetch(`${API_URL}/api/chat`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        session_id,
                        user_id,
                        messages: messages_history,
                        top_k: 10,
                        use_reranker: false
                    }),
            }  
        );

        if (!data.ok) throw new Error();

        const {answer} = await data.json();
        return answer;
    }


