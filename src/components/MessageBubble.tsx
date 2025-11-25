"use client";

import Image from "next/image";
import dynamic from "next/dynamic";

const MoleculeViewer = dynamic(() => import("./MoleculeViewer"), {
	ssr: false,
	loading: () => (
		<div className="w-full h-64 bg-gray-800 rounded-lg animate-pulse flex items-center justify-center text-gray-500">
			Loading 3D Viewer...
		</div>
	),
});

export type Message = {
	id: string;
	role: "user" | "assistant";
	content: string;
	image?: string;
	moleculeData?: {
		name: string;
		info: string;
		structure?: string; // SDF or PDB string
		format?: "sdf" | "pdb";
	};
};

interface MessageBubbleProps {
	message: Message;
}

export default function MessageBubble({ message }: MessageBubbleProps) {
	const isUser = message.role === "user";

	return (
		<div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
			<div
				className={`max-w-[85%] rounded-2xl p-4 ${
					isUser
						? "bg-blue-600 text-white rounded-br-none"
						: "bg-gray-800 text-gray-100 rounded-bl-none border border-gray-700"
				}`}
			>
				{/* User Uploaded Image */}
				{message.image && (
					<div className="mb-3 relative h-48 w-full min-w-[200px] rounded-lg overflow-hidden bg-black/20">
						<Image src={message.image} alt="Uploaded molecule" fill className="object-contain" />
					</div>
				)}

				{/* Text Content */}
				<p className="whitespace-pre-wrap leading-relaxed mb-2">{message.content}</p>

				{/* AI Analysis Result */}
				{message.moleculeData && (
					<div className="mt-4 space-y-3">
						<div className="bg-gray-900/50 p-3 rounded-lg border border-gray-600">
							<h3 className="font-bold text-blue-300 text-lg">{message.moleculeData.name}</h3>
							<p className="text-sm text-gray-300 mt-1">{message.moleculeData.info}</p>
						</div>

						{/* 3D Viewer */}
						{message.moleculeData.structure && (
							<div className="mt-2">
								<p className="text-xs text-gray-400 mb-1 uppercase font-semibold tracking-wider">3D Structure</p>
								<MoleculeViewer
									sdf={message.moleculeData.format === "sdf" ? message.moleculeData.structure : undefined}
									pdb={message.moleculeData.format === "pdb" ? message.moleculeData.structure : undefined}
									format={message.moleculeData.format}
								/>
							</div>
						)}
					</div>
				)}
			</div>
		</div>
	);
}
