export function sendMessage(
  element: HTMLButtonElement,
  inputElement: HTMLInputElement,
  chatHistoryElement: HTMLDivElement,
  measureSpan: HTMLSpanElement
) {
  element.addEventListener("click", async () => {
    const message = inputElement.value.trim();
    if (!message) return; // Don't send empty messages

    // Create new chat entry container
    const chatEntry = document.createElement("div");
    chatEntry.classList.add("chat-entry");

    // Add user message immediately
    chatEntry.innerHTML = `<div class="history-chat-input">${message}</div>`;

    try {
      // Wait for backend response
      const response = await fetchData(message);

      // Append backend response once received
      const responseElement = document.createElement("div");
      responseElement.classList.add("history-chat-output");
      responseElement.textContent = response;
      chatEntry.appendChild(responseElement);
    } catch (error) {
      console.error("Error fetching message:", error);
    }

    chatHistoryElement.appendChild(chatEntry);

    // Clear input field
    inputElement.value = "";

    // Reset input width
    adjustInputWidth(inputElement, measureSpan);

    // Auto-scroll to latest chat entry
    chatHistoryElement.scrollTop = chatHistoryElement.scrollHeight;
    window.scrollTo({ top: document.documentElement.scrollHeight, behavior: "instant" });
  });
}


// Adjust input width dynamically
export function adjustInputWidth(input: HTMLInputElement, measureSpan: HTMLSpanElement) {
  measureSpan.textContent = input.value || input.placeholder;
  measureSpan.style.font = window.getComputedStyle(input).font;
  input.style.width = `${measureSpan.offsetWidth-15}px`; // Adjust for padding
}

import axios from "axios";

const API_URL = "http://127.0.0.1:8000"; // Python Backend URL

export const fetchData = async (user_message: string) => {
  try {
    const response = await axios.post(`${API_URL}/api/data`, {
      message: user_message,
    });
    console.log("Response fetched:", response);
    console.log("Data fetched:", response.data);
    console.log("message fetched:", response.data.message);
    return response.data.message;
  } catch (error) {
    console.error("Error fetching data:", error);
    throw error;
  }
};
