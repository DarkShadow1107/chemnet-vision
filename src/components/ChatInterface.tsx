"use client";

import { useState, useRef, useEffect } from "react";
import MessageBubble, { Message } from "./MessageBubble";

type Conversation = {
	id: string;
	title: string;
	messages: Message[];
	updatedAt: string;
};

export default function ChatInterface() {
	const [conversations, setConversations] = useState<Conversation[]>([]);
	const [currentConversationId, setCurrentConversationId] = useState<string | null>(null);
	const [messages, setMessages] = useState<Message[]>([]);
	const [input, setInput] = useState("");
	const [loading, setLoading] = useState(false);
	const [isSidebarOpen, setIsSidebarOpen] = useState(false);
	const fileInputRef = useRef<HTMLInputElement>(null);
	const messagesEndRef = useRef<HTMLDivElement>(null);

	const scrollToBottom = () => {
		messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
	};

	// Load conversations on mount
	useEffect(() => {
		fetchConversations();
	}, []);

	// Save conversation whenever messages change
	useEffect(() => {
		if (messages.length > 0 && currentConversationId) {
			saveConversation(currentConversationId, messages);
		}
	}, [messages, currentConversationId]);

	useEffect(() => {
		scrollToBottom();
	}, [messages, loading]);

	const fetchConversations = async () => {
		try {
			const res = await fetch("/api/conversations");
			const data = await res.json();
			setConversations(data);

			// If no current conversation, start a new one or load the last one
			if (!currentConversationId && data.length > 0) {
				// Optional: Load the most recent one
				// loadConversation(data[data.length - 1]);
			}
		} catch (error) {
			console.error("Failed to fetch conversations", error);
		}
	};

	const saveConversation = async (id: string, msgs: Message[]) => {
		try {
			// Determine title from first user message if it's a new conversation
			let title = "New Conversation";
			const firstUserMsg = msgs.find((m) => m.role === "user");
			if (firstUserMsg) {
				title = firstUserMsg.content.slice(0, 30) + (firstUserMsg.content.length > 30 ? "..." : "");
			}

			await fetch("/api/conversations", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ id, messages: msgs, title }),
			});

			// Refresh list to update titles/order
			fetchConversations();
		} catch (error) {
			console.error("Failed to save conversation", error);
		}
	};

	const startNewConversation = () => {
		const newId = Date.now().toString();
		setCurrentConversationId(newId);
		setMessages([
			{
				id: "init",
				role: "assistant",
				content:
					"Hello! I am ChemNet-Vision. I can analyze molecule images or answer questions about chemical compounds. Try asking about 'Aspirin' or upload a 2D structure!",
			},
		]);
		setIsSidebarOpen(false);
	};

	const loadConversation = (conv: Conversation) => {
		setCurrentConversationId(conv.id);
		setMessages(conv.messages);
		setIsSidebarOpen(false);
	};

	// Initialize with a new conversation if none exists
	useEffect(() => {
		if (!currentConversationId) {
			startNewConversation();
		}
	}, []);

	const handleSendMessage = async () => {
		if (!input.trim()) return;

		const newMessage: Message = {
			id: Date.now().toString(),
			role: "user",
			content: input,
		};

		setMessages((prev) => [...prev, newMessage]);
		setInput("");
		setLoading(true);

		try {
			const response = await fetch("http://localhost:5000/chat", {
				method: "POST",
				headers: {
					"Content-Type": "application/json",
				},
				body: JSON.stringify({ message: input }),
			});
			const data = await response.json();

			const aiResponse: Message = {
				id: (Date.now() + 1).toString(),
				role: "assistant",
				content: data.content,
				image: data.image,
				moleculeData: data.moleculeData,
			};
			setMessages((prev) => [...prev, aiResponse]);
		} catch (error) {
			console.error("Error sending message:", error);
			setMessages((prev) => [
				...prev,
				{
					id: (Date.now() + 1).toString(),
					role: "assistant",
					content: "Sorry, I encountered an error connecting to the server. Please ensure the backend is running.",
				},
			]);
		} finally {
			setLoading(false);
		}
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
					content: data.error ? `Error: ${data.error}` : "Here is what I found based on the image analysis:",
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
		<div className="flex h-full relative bg-[radial-gradient(ellipse_at_top,var(--tw-gradient-stops))] from-slate-900 via-[#0f172a] to-black overflow-hidden">
			{/* Sidebar Toggle (Mobile) */}
			<button
				onClick={() => setIsSidebarOpen(!isSidebarOpen)}
				className="absolute top-4 left-4 z-50 p-2 bg-white/10 rounded-full hover:bg-white/20 transition-colors md:hidden"
			>
				<svg
					xmlns="http://www.w3.org/2000/svg"
					fill="none"
					viewBox="0 0 24 24"
					strokeWidth={1.5}
					stroke="currentColor"
					className="w-6 h-6 text-white"
				>
					<path strokeLinecap="round" strokeLinejoin="round" d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25h16.5" />
				</svg>
			</button>

			{/* Sidebar */}
			<div
				className={`
                fixed md:relative z-40 h-full w-72 bg-black/50 backdrop-blur-xl border-r border-white/10 transform transition-transform duration-300 ease-in-out
                ${isSidebarOpen ? "translate-x-0" : "-translate-x-full md:translate-x-0"}
            `}
			>
				<div className="p-4 flex flex-col h-full">
					<button
						onClick={startNewConversation}
						className="w-full py-3 px-4 bg-blue-600 hover:bg-blue-500 text-white rounded-xl font-medium transition-colors flex items-center justify-center gap-2 mb-6 shadow-lg shadow-blue-900/20"
					>
						<svg
							xmlns="http://www.w3.org/2000/svg"
							fill="none"
							viewBox="0 0 24 24"
							strokeWidth={1.5}
							stroke="currentColor"
							className="w-5 h-5"
						>
							<path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m7.5-7.5h-15" />
						</svg>
						New Chat
					</button>

					<div className="flex-1 overflow-y-auto space-y-2 pr-2 custom-scrollbar">
						<h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3 px-2">Recent</h3>
						{conversations
							.slice()
							.reverse()
							.map((conv) => (
								<button
									key={conv.id}
									onClick={() => loadConversation(conv)}
									className={`w-full text-left p-3 rounded-lg text-sm transition-all duration-200 truncate ${
										currentConversationId === conv.id
											? "bg-white/10 text-white shadow-inner"
											: "text-slate-400 hover:bg-white/5 hover:text-slate-200"
									}`}
								>
									{conv.title}
								</button>
							))}
					</div>
				</div>
			</div>

			{/* Chat Area */}
			<div className="flex-1 flex flex-col h-full relative w-full">
				<div className="flex-1 overflow-y-auto pt-24 pb-32 px-4 sm:px-6 scroll-smooth">
					<div className="max-w-4xl mx-auto space-y-8">
						{messages.map((msg) => (
							<MessageBubble key={msg.id} message={msg} />
						))}

						{loading && (
							<div className="flex justify-start animate-fade-in">
								<div className="glass px-6 py-4 rounded-2xl rounded-bl-sm flex items-center gap-3">
									<span className="text-sm text-slate-400 font-medium">Thinking</span>
									<div className="flex gap-1.5">
										<div
											className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-bounce"
											style={{ animationDelay: "0ms" }}
										/>
										<div
											className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-bounce"
											style={{ animationDelay: "150ms" }}
										/>
										<div
											className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-bounce"
											style={{ animationDelay: "300ms" }}
										/>
									</div>
								</div>
							</div>
						)}
						<div ref={messagesEndRef} />
					</div>
				</div>

				{/* Input Area */}
				<div className="absolute bottom-6 left-0 right-0 px-4 z-20">
					<div className="max-w-3xl mx-auto">
						<div className="glass p-2 rounded-3xl shadow-2xl shadow-black/50 border border-white/10 flex items-end gap-2 backdrop-blur-xl">
							<button
								onClick={() => fileInputRef.current?.click()}
								className="p-3 text-slate-400 hover:text-blue-400 hover:bg-white/5 rounded-full transition-all duration-200"
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
										d="m2.25 15.75 5.159-5.159a2.25 2.25 0 0 1 3.182 0l5.159 5.159m-1.5-1.5 1.409-1.409a2.25 2.25 0 0 1 3.182 0l2.909 2.909m-18 3.75h16.5a1.5 1.5 0 0 0 1.5-1.5V6a1.5 1.5 0 0 0-1.5-1.5H3.75A1.5 1.5 0 0 0 2.25 6v12a1.5 1.5 0 0 0 1.5 1.5Zm10.5-11.25h.008v.008h-.008V8.25Zm.375 0a.375.375 0 1 1-.75 0 .375.375 0 0 1 .75 0Z"
									/>
								</svg>
							</button>
							<input
								type="file"
								ref={fileInputRef}
								onChange={handleFileUpload}
								accept="image/*"
								className="hidden"
							/>

							<div className="flex-1 relative">
								<textarea
									value={input}
									onChange={(e) => setInput(e.target.value)}
									onKeyDown={handleKeyDown}
									placeholder="Ask about a molecule or upload an image..."
									className="w-full bg-transparent text-white p-3 max-h-32 min-h-12 resize-none focus:outline-none placeholder:text-slate-500"
									rows={1}
									style={{ height: "auto", minHeight: "48px" }}
								/>
							</div>

							<button
								onClick={handleSendMessage}
								disabled={!input.trim()}
								className={`p-3 rounded-full transition-all duration-200 ${
									input.trim()
										? "bg-blue-600 text-white hover:bg-blue-500 shadow-lg shadow-blue-600/20 transform hover:scale-105"
										: "bg-white/5 text-slate-600 cursor-not-allowed"
								}`}
							>
								<svg
									xmlns="http://www.w3.org/2000/svg"
									viewBox="0 0 24 24"
									fill="currentColor"
									className="w-5 h-5"
								>
									<path d="M3.478 2.404a.75.75 0 0 0-.926.941l2.432 7.905H13.5a.75.75 0 0 1 0 1.5H4.984l-2.432 7.905a.75.75 0 0 0 .926.94 60.519 60.519 0 0 0 18.445-8.986.75.75 0 0 0 0-1.218A60.517 60.517 0 0 0 3.478 2.404Z" />
								</svg>
							</button>
						</div>
						<p className="text-center text-xs text-slate-600 mt-3">
							AI can make mistakes. Please verify important chemical information.
						</p>
					</div>
				</div>
			</div>
		</div>
	);
}
