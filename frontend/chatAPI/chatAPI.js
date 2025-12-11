
        const API_URL = 'http://127.0.0.1:8000';
         import { nanoid } from "https://cdn.jsdelivr.net/npm/nanoid/+esm";

        
        let session_id = JSON.parse(sessionStorage.getItem('session_id'));

        if (!session_id) {
            session_id = nanoid();
            sessionStorage.setItem('session_id', JSON.stringify(session_id));
        }


        export async function sendMessageToChat (text) {
                const data  = await fetch(`${API_URL}/api/chat`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        session_id,
                        messages : [
                            {role: 'user', content: text}
                        ],
                        top_k: 10,
                        use_reranker: false
                    }),
        }  
        );

        if (!data.ok) throw new Error();

        const {answer} = await data.json();
        return answer;
    }