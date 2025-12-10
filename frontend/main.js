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

        let isLoading = false;
        const session_id = JSON.parse(sessionStorage.getItem('session_id'));
        const messages_history = JSON.parse(sessionStorage.getItem('messages-history')) || [];
        const user_id = JSON.parse(localStorage.getItem('user_id'));


        if(!user_id) {
            localStorage.setItem('user_id', JSON.stringify(nanoid()));
        }

        if (!session_id) {
            sessionStorage.setItem('session_id', JSON.stringify(nanoid()));
        }

        let isFirstOpen = true;


        //Open chatbot
        chatbotToggle.addEventListener('click', () => {
            chatbotContainer.classList.add('open');
            chatbotToggle.classList.add('hidden');

            if (messages_history.length && isFirstOpen) {
                getChatHistory();
            } else if (isFirstOpen) {
                addMessageToChat(GREETING_MESSAGE, 'assistant');
                setMessageToStorage(GREETING_MESSAGE, 'assistant');
            }

            isFirstOpen = false;
            chatbotInput.focus();
        });

        //Close chatbot
        chatbotClose.addEventListener('click', () => {
            chatbotContainer.classList.remove('open');
            chatbotToggle.classList.remove('hidden');
        });

        //Check input value and disable/enable send button
        chatbotInput.addEventListener('input', onInputChange);

        //Send message on button click or enter key press
        chatbotSend.addEventListener('click', sendMessage);

        chatbotInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !isLoading && !chatbotSend.disabled) sendMessage();
        });


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
            isLoading = true; //Turn on loading 
            chatbotSend.innerHTML = "<span class='loader'></span>" //Show loader in the send button
            const text = chatbotInput.value.trim();
            addMessageToChat(text, 'user');
            setMessageToStorage(text, 'user');
            setDisabled(chatbotSend, true)   //Disable user input and send button during loading
            setDisabled(chatbotInput, true)
            

            let content;

            try {
            const answer = await sendMessageToChat();
            content = answer;
            // setMessageToStorage(answer, 'assistant');
            // addMessageToChat(answer, 'assistant');
            chatbotInput.value = ''; //Clear input value if all good

         }catch (error) {
            setDisabled(chatbotSend, false);
            content = ERROR_MESSAGE;
                
            } finally {
                isLoading = false;
                setMessageToStorage(content, 'assistant');
                addMessageToChat(content, 'assistant');
                chatbotSend.innerHTML = 'Send'
                setDisabled(chatbotInput, false)
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

