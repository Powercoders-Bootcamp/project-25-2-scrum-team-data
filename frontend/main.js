    
        import { sendMessageToChat } from "./chatAPI/chatAPI.js";

        const chatbotToggle = document.getElementById('chatbotToggle');
        const chatbotContainer = document.getElementById('chatbotContainer');
        const chatbotClose = document.getElementById('chatbotClose');
        const chatbotMessages = document.getElementById('chatbotMessages');
        const chatbotInput = document.getElementById('chatbotInput');
        const chatbotSend = document.getElementById('chatbotSend');

        const GREETING_MESSAGE = "Hello, I'm an AI assistant. How can I help you?";
        const ERROR_MESSAGE = "Sorry, something went wrong while processing your request.";
        const REPEAT_GREATING_MESSAGE = "Welcome back! HoI remember everething we talked about last time. How can I assist you further?";

        let isLoading = false;
        let isFirstOpen = true;

        const session_id = JSON.parse(sessionStorage.getItem('session_id'));

        //Open chatbot
        chatbotToggle.addEventListener('click', () => {
            chatbotContainer.classList.add('open');
            chatbotToggle.classList.add('hidden');

            if ( isFirstOpen && session_id) {
                addMessageToChat(REPEAT_GREATING_MESSAGE, 'assistant');
            } else if (isFirstOpen) {
                addMessageToChat(GREETING_MESSAGE, 'assistant');
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
            setDisabled(chatbotSend, true)   //Disable user input and send button during loading
            setDisabled(chatbotInput, true)

            let content;

            try {

            const check = await sendMessageToChat(chatbotInput.value.trim()); 
            content = check;
            console.log('object :>> ', check);
            chatbotInput.value = ''; //Clear input value if all good

         }catch (error) {
            setDisabled(chatbotSend, false);
            content = ERROR_MESSAGE;
                
            } finally {
                isLoading = false;
                addMessageToChat(content, 'assistant');
                chatbotSend.innerHTML = 'Send'
                setDisabled(chatbotInput, false)
            }
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



