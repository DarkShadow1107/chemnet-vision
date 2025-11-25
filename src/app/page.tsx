"use client";

import ChatInterface from "@/components/ChatInterface";

export default function Home() {
	return (
		<main className="flex h-screen flex-col bg-gray-900 text-white">
			{/* Header */}
			<header className="p-4 border-b border-gray-800 bg-gray-900/50 backdrop-blur-md fixed w-full top-0 z-10">
				<div className="max-w-4xl mx-auto flex items-center gap-3">
					<div className="w-8 h-8 rounded-full bg-linear-to-r from-blue-500 to-purple-600 flex items-center justify-center font-bold text-sm">
						CV
					</div>
					<h1 className="text-xl font-bold bg-clip-text text-transparent bg-linear-to-r from-blue-400 to-purple-600">
						ChemNet-Vision
					</h1>
				</div>
			</header>

			<ChatInterface />
		</main>
	);
}
