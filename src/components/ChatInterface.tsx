"use client";

import { useState, useRef, useEffect } from "react";
import MessageBubble, { Message } from "./MessageBubble";

export default function ChatInterface() {
	const [messages, setMessages] = useState<Message[]>([
		{
			id: "1",
			role: "assistant",
			content:
				"Hello! I am ChemNet-Vision. Upload a 2D molecule image, and I will identify it and show you its 3D structure.",
		},
	]);
	const [input, setInput] = useState("");
	const [loading, setLoading] = useState(false);
	const fileInputRef = useRef<HTMLInputElement>(null);
	const messagesEndRef = useRef<HTMLDivElement>(null);

	const scrollToBottom = () => {
		messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
	};

	useEffect(() => {
		scrollToBottom();
	}, [messages]);

	const handleSendMessage = async () => {
		if (!input.trim()) return;

		const newMessage: Message = {
			id: Date.now().toString(),
			role: "user",
			content: input,
		};

		setMessages((prev) => [...prev, newMessage]);
		setInput("");

		// Simulate AI response for text only
		setTimeout(() => {
			setMessages((prev) => [
				...prev,
				{
					id: (Date.now() + 1).toString(),
					role: "assistant",
					content: "I am currently tuned to analyze images. Please upload a molecule image!",
				},
			]);
		}, 1000);
	};

	const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
		if (e.target.files && e.target.files[0]) {
			const file = e.target.files[0];
			const imageUrl = URL.createObjectURL(file);

			const userMessage: Message = {
				id: Date.now().toString(),
				role: "user",
				content: "Analyze this molecule:",
				image: imageUrl,
			};

			setMessages((prev) => [...prev, userMessage]);
			setLoading(true);

			const formData = new FormData();
			formData.append("file", file);

			try {
				const response = await fetch("http://localhost:5000/predict", {
					method: "POST",
					body: formData,
				});
				const data = await response.json();

				const aiResponse: Message = {
					id: (Date.now() + 1).toString(),
					role: "assistant",
					content: data.error ? `Error: ${data.error}` : "Here is what I found:",
					moleculeData: !data.error
						? {
								name: data.molecule,
								info: data.info,
								structure: data.structure, // Expecting SDF/PDB string from backend
								format: "sdf", // Defaulting to SDF for now
						  }
						: undefined,
				};
				setMessages((prev) => [...prev, aiResponse]);
			} catch (error) {
				console.error("Error uploading file:", error);
				setMessages((prev) => [
					...prev,
					{
						id: (Date.now() + 1).toString(),
						role: "assistant",
						content: "Sorry, I encountered an error processing your image.",
					},
				]);
			} finally {
				setLoading(false);
				if (fileInputRef.current) fileInputRef.current.value = "";
			}
		}
	};

	const handleKeyDown = (e: React.KeyboardEvent) => {
		if (e.key === "Enter" && !e.shiftKey) {
			e.preventDefault();
			handleSendMessage();
		}
	};

	return (
		<div className="flex flex-col h-full">
			{/* Chat Area */}
			<div className="flex-1 overflow-y-auto pt-20 pb-24 px-4">
				<div className="max-w-4xl mx-auto space-y-6">
					{messages.map((msg) => (
						<MessageBubble key={msg.id} message={msg} />
					))}

					{loading && (
						<div className="flex justify-start">
							<div className="bg-gray-800 rounded-2xl rounded-bl-none p-4 border border-gray-700 flex items-center gap-2">
								<div
									className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"
									style={{ animationDelay: "0ms" }}
								/>
								<div
									className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"
									style={{ animationDelay: "150ms" }}
								/>
								<div
									className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"
									style={{ animationDelay: "300ms" }}
								/>
							</div>
						</div>
					)}
					<div ref={messagesEndRef} />
				</div>
			</div>

			{/* Input Area */}
			<div className="fixed bottom-0 w-full bg-gray-900 border-t border-gray-800 p-4">
				<div className="max-w-4xl mx-auto flex items-end gap-3">
					<button
						onClick={() => fileInputRef.current?.click()}
						className="p-3 text-gray-400 hover:text-blue-400 hover:bg-gray-800 rounded-full transition-colors"
						title="Upload Image"
					>
						<svg
							xmlns="http://www.w3.org/2000/svg"
							fill="none"
							viewBox="0 0 24 24"
							strokeWidth={1.5}
							stroke="currentColor"
							className="w-6 h-6"
						>
							<path
								strokeLinecap="round"
								strokeLinejoin="round"
								d="m18.375 12.739-7.693 7.693a4.5 4.5 0 0 1-6.364-6.364l10.94-10.94A3 3 0 1 1 19.5 7.372L8.552 18.32m.009-.01-.01.01m5.699-9.941-7.81 7.81a1.5 1.5 0 0 0 2.112 2.13"
							/>
						</svg>
					</button>
					<input type="file" ref={fileInputRef} onChange={handleFileUpload} accept="image/*" className="hidden" />

					<div className="flex-1 bg-gray-800 rounded-2xl border border-gray-700 focus-within:border-blue-500 focus-within:ring-1 focus-within:ring-blue-500 transition-all">
						<textarea
							value={input}
							onChange={(e) => setInput(e.target.value)}
							onKeyDown={handleKeyDown}
							placeholder="Type a message..."
							className="w-full bg-transparent text-white p-3 max-h-32 min-h-12 resize-none focus:outline-none"
							rows={1}
						/>
					</div>

					<button
						onClick={handleSendMessage}
						disabled={!input.trim()}
						className={`p-3 rounded-full transition-all ${
							input.trim()
								? "bg-blue-600 text-white hover:bg-blue-700 shadow-lg shadow-blue-600/20"
								: "bg-gray-800 text-gray-500 cursor-not-allowed"
						}`}
					>
						<svg
							xmlns="http://www.w3.org/2000/svg"
							fill="none"
							viewBox="0 0 24 24"
							strokeWidth={1.5}
							stroke="currentColor"
							className="w-6 h-6"
						>
							<path
								strokeLinecap="round"
								strokeLinejoin="round"
								d="M6 12 3.269 3.126A59.768 59.768 0 0 1 21.485 12 59.77 59.77 0 0 1 3.27 20.876L5.999 12Zm0 0h7.5"
							/>
						</svg>
					</button>
				</div>
			</div>
		</div>
	);
}
