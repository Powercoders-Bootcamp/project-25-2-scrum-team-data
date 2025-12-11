        import { nanoid } from 'https://cdn.jsdelivr.net/npm/nanoid/+esm';
    
        import { sendMessageToChat } from "./chatAPI/chatAPI.js";

        const chatbotToggle = document.getElementById('chatbotToggle');
        const chatbotContainer = document.getElementById('chatbotContainer');
        const chatbotClose = document.getElementById('chatbotClose');
        const chatbotMessages = document.getElementById('chatbotMessages');
        const chatbotInput = document.getElementById('chatbotInput');
        const chatbotSend = document.getElementById('chatbotSend');

        const GREETING_MESSAGE = "Hello, I'm an AI assistant. How can I help you?";
        const ERROR_MESSAGE = "Sorry, something went wrong while processing your request.";

        console.log('🚀 Chatbot initialized');
        
        let isLoading = false;
        
        // Initialize user_id and session_id if they don't exist
        let user_id, session_id, messages_history;
        
        try {
            const stored_user_id = localStorage.getItem('user_id');
            user_id = stored_user_id ? JSON.parse(stored_user_id) : null;
        } catch (e) {
            console.error('Error parsing user_id:', e);
            user_id = null;
        }
        
        try {
            const stored_session_id = sessionStorage.getItem('session_id');
            session_id = stored_session_id ? JSON.parse(stored_session_id) : null;
        } catch (e) {
            console.error('Error parsing session_id:', e);
            session_id = null;
        }
        
        try {
            const stored_messages = sessionStorage.getItem('messages-history');
            messages_history = stored_messages ? JSON.parse(stored_messages) : [];
        } catch (e) {
            console.error('Error parsing messages_history:', e);
            messages_history = [];
        }

        if(!user_id) {
            user_id = nanoid();
            localStorage.setItem('user_id', JSON.stringify(user_id));
            console.log('✅ Created new user_id:', user_id);
        } else {
            console.log('✅ Using existing user_id:', user_id);
        }

        if (!session_id) {
            session_id = nanoid();
            sessionStorage.setItem('session_id', JSON.stringify(session_id));
            console.log('✅ Created new session_id:', session_id);
        } else {
            console.log('✅ Using existing session_id:', session_id);
        }

        let isFirstOpen = true;
        let allowClose = false; // Sadece closeChatbot() çağrıldığında true olur

        // Chatbot container'ın classList.remove metodunu override et
        // Böylece 'open' class'ı sadece izin verildiğinde kaldırılabilir
        const originalRemove = chatbotContainer.classList.remove.bind(chatbotContainer.classList);
        chatbotContainer.classList.remove = function(...args) {
            // Eğer 'open' class'ı kaldırılmaya çalışılıyorsa ve izin yoksa, engelle
            if (args.includes('open') && !allowClose) {
                console.warn('⚠️ Attempted to close chatbot without permission. Blocked.');
                return; // Kapanmayı engelle
            }
            // Diğer class'lar için normal davranış
            return originalRemove(...args);
        };

        // Chatbot'un açık kalmasını garanti etmek için periyodik kontrol
        setInterval(() => {
            // Eğer chatbot açık olması gerekiyorsa ama kapalıysa, tekrar aç
            if (!allowClose && !chatbotContainer.classList.contains('open') && chatbotToggle.classList.contains('hidden')) {
                console.log('⚠️ Chatbot was closed unexpectedly, reopening...');
                chatbotContainer.classList.add('open');
            }
        }, 500); // Her 500ms'de bir kontrol et

        //Open chatbot
        chatbotToggle.addEventListener('click', () => {
            console.log('👆 Chatbot toggle clicked');
            openChatbot();
        });
        
        // Chatbot'u açma fonksiyonu
        function openChatbot() {
            chatbotContainer.classList.add('open');
            chatbotToggle.classList.add('hidden');
            allowClose = false; // Açıldığında kapanmaya izin verme

            if (messages_history.length && isFirstOpen) {
                console.log('📜 Loading chat history');
                getChatHistory();
            } else if (isFirstOpen) {
                console.log('👋 Showing greeting message');
                addMessageToChat(GREETING_MESSAGE, 'assistant');
                setMessageToStorage(GREETING_MESSAGE, 'assistant');
            }

            isFirstOpen = false;
            chatbotInput.focus();
        }

        //Close chatbot - SADECE X butonuna basıldığında kapat
        chatbotClose.addEventListener('click', (e) => {
            e.stopPropagation(); // Event propagation'ı durdur
            console.log('❌ Chatbot close clicked');
            closeChatbot();
        });
        
        // Chatbot'u kapatma fonksiyonu - sadece bu fonksiyon chatbot'u kapatabilir
        function closeChatbot() {
            allowClose = true; // Kapanmaya izin ver
            chatbotContainer.classList.remove('open');
            chatbotToggle.classList.remove('hidden');
            // İzin flag'ini hemen sıfırla
            setTimeout(() => {
                allowClose = false;
            }, 100);
        }
        
        // Chatbot container içindeki tıklamaların dışarı çıkmasını engelle
        // Ama sadece container'ın kendisine tıklandığında (içindeki elementlere değil)
        chatbotContainer.addEventListener('click', (e) => {
            // Eğer tıklama container'ın kendisine ise (içindeki bir elemente değilse)
            if (e.target === chatbotContainer) {
                e.stopPropagation();
            }
        });

        //Check input value and disable/enable send button
        chatbotInput.addEventListener('input', onInputChange);

        //Send message on button click or enter key press
        chatbotSend.addEventListener('click', () => {
            console.log('👆 Send button clicked');
            sendMessage();
        });

        chatbotInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !isLoading && !chatbotSend.disabled) {
                console.log('⌨️ Enter key pressed');
                sendMessage();
            }
        });
        
        console.log('✅ Event listeners attached');


        function addMessageToChat (markup, sender) {
            chatbotMessages.innerHTML += makeMessageMarkup(markup, sender);
            chatbotMessages.scrollTop = chatbotMessages.scrollHeight;

        }


        function makeMessageMarkup (message, sender) {          
             return `<div class="message ${sender}">
            <p class="message-label">${sender}</p>
            <div class="message-content">${message}</div>
            </div>`
        }


        async function sendMessage() {
            console.log('📤 sendMessage called');
            isLoading = true; //Turn on loading 
            chatbotSend.innerHTML = "<span class='loader'></span>" //Show loader in the send button
            const text = chatbotInput.value.trim();
            console.log('📝 User message:', text);
            
            if (!text) {
                console.warn('⚠️ Empty message, aborting');
                isLoading = false;
                return;
            }
            
            addMessageToChat(text, 'user');
            setMessageToStorage(text, 'user');
            setDisabled(chatbotSend, true)   //Disable user input and send button during loading
            setDisabled(chatbotInput, true)
            

            let content;

            try {
            console.log('🔄 Calling sendMessageToChat...');
            const answer = await sendMessageToChat();
            console.log('✅ Received answer:', answer);
            content = answer;
            chatbotInput.value = ''; //Clear input value if all good

         }catch (error) {
            console.error('❌ Error sending message:', error);
            console.error('Error details:', {
                message: error.message,
                name: error.name,
                stack: error.stack
            });
            setDisabled(chatbotSend, false);
            // Show more specific error message if available
            content = error.message || ERROR_MESSAGE;
                
            } finally {
                isLoading = false;
                setMessageToStorage(content, 'assistant');
                addMessageToChat(content, 'assistant');
                chatbotSend.innerHTML = 'Send'
                setDisabled(chatbotInput, false)
                
                // Chatbot'un açık kalmasını garanti et - sadece X butonuna basıldığında kapanmalı
                // Eğer kapanmışsa tekrar aç (allowClose kontrolü ile)
                if (!allowClose && !chatbotContainer.classList.contains('open')) {
                    console.log('⚠️ Chatbot was closed unexpectedly, reopening...');
                    chatbotContainer.classList.add('open');
                    chatbotToggle.classList.add('hidden');
                }
                
                // Input'a focus ver ki kullanıcı hemen yeni mesaj yazabilsin
                setTimeout(() => {
                    chatbotInput.focus();
                }, 100);
                
                console.log('✅ Message processing completed');
            }
        }

        function getChatHistory() {
                chatbotMessages.innerHTML = makeHistoryMarkup();
                chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
        }


        function onInputChange({target : {value}}) {
            if (!value.trim()) {
                setDisabled(chatbotSend, true)
            } else if (!isLoading) {
                setDisabled(chatbotSend, false)
            }
        }

        function setDisabled(element, isDisabled) {
            isDisabled ? element.setAttribute('disabled', true) : element.removeAttribute('disabled');
        }


        function setMessageToStorage(content, role) {
            messages_history.push({role, content});
            sessionStorage.setItem('messages-history', JSON.stringify(messages_history))  
        }

        function makeHistoryMarkup() {
            const markup = messages_history.reduce((acc, {role, content}) => acc += makeMessageMarkup(content, role), '')   
            return markup     
        }

