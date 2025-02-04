// Refactored sendMessage function
export function sendMessage(
  element: HTMLButtonElement,
  inputElement: HTMLInputElement,
  chatHistoryElement: HTMLDivElement,
  measureSpan: HTMLSpanElement
) {
  element.addEventListener('click', () => {
    const message = inputElement.value.trim();
    if (!message) return; // Don't send empty messages

    console.log(message);

    // Create new chat entry
    const chatEntry = document.createElement('div');
    chatEntry.classList.add('chat-entry');
    chatEntry.innerHTML = `
      <div class="history-chat-input">${message}</div>
      <div class="history-chat-output">${message}</div>
    `;

    // Append chat entry to chat history
    chatHistoryElement.appendChild(chatEntry);

    // Clear input field
    inputElement.value = '';

    // Reset input width
    adjustInputWidth(inputElement, measureSpan);

    chatHistoryElement.scrollTop = chatHistoryElement.scrollHeight;

    window.scrollTo({
      top: document.documentElement.scrollHeight,
      behavior: "instant",
    });
  });
}

// Adjust input width dynamically
export function adjustInputWidth(input: HTMLInputElement, measureSpan: HTMLSpanElement) {
  measureSpan.textContent = input.value || input.placeholder;
  measureSpan.style.font = window.getComputedStyle(input).font;
  input.style.width = `${measureSpan.offsetWidth}px`; // Adjust for padding
}
