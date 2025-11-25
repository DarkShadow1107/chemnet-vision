"use client";

import Image from "next/image";
import dynamic from "next/dynamic";

const MoleculeViewer = dynamic(() => import("./MoleculeViewer"), {
	ssr: false,
	loading: () => (
		<div className="w-full h-64 bg-slate-800/50 rounded-xl animate-pulse flex items-center justify-center text-slate-500 border border-slate-700/50">
			<div className="flex flex-col items-center gap-2">
				<svg className="w-6 h-6 animate-spin" fill="none" viewBox="0 0 24 24">
					<circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
					<path
						className="opacity-75"
						fill="currentColor"
						d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
					></path>
				</svg>
				<span className="text-sm font-medium">Loading 3D Viewer...</span>
			</div>
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
		<div className={`flex w-full ${isUser ? "justify-end" : "justify-start"} animate-fade-in group`}>
			<div
				className={`relative max-w-[85%] lg:max-w-[75%] rounded-2xl p-5 shadow-lg transition-all duration-200 ${
					isUser
						? "bg-linear-to-br from-teal-800 to-cyan-900 text-white rounded-br-sm shadow-teal-900/20"
						: "bg-slate-950/70 backdrop-blur-md text-slate-100 rounded-bl-sm border border-white/5 shadow-black/20"
				}`}
			>
				{/* Role Label (Optional, maybe just for Assistant) */}
				{!isUser && (
					<div className="absolute -top-6 left-0 text-xs font-medium text-slate-500 flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
						<div className="w-4 h-4 rounded-full bg-linear-to-r from-blue-600 to-purple-600 flex items-center justify-center text-[8px] text-white font-bold shadow-lg">
							CV
						</div>
						ChemNet-Vision
					</div>
				)}

				{/* User Uploaded Image */}
				{message.image && (
					<div className="mb-4 relative h-56 w-full min-w-60 rounded-xl overflow-hidden bg-black/40 border border-white/5">
						<Image src={message.image} alt="Uploaded molecule" fill className="object-contain p-2" />
					</div>
				)}

				{/* Text Content */}
				<div className="prose prose-invert max-w-none">
					<p className="whitespace-pre-wrap leading-relaxed text-[15px]">{message.content}</p>
				</div>

				{/* AI Analysis Result */}
				{message.moleculeData && (
					<div className="mt-5 space-y-4">
						<div className="bg-slate-900/50 p-4 rounded-xl border-l-4 border-l-blue-500 border-y border-r border-white/5">
							<h3 className="font-bold text-blue-300 text-lg flex items-center gap-2">
								<svg
									xmlns="http://www.w3.org/2000/svg"
									viewBox="0 0 24 24"
									fill="currentColor"
									className="w-5 h-5"
								>
									<path
										fillRule="evenodd"
										d="M14.615 1.595a.75.75 0 01.359.852L12.982 9.75h7.268a.75.75 0 01.548 1.262l-10.5 11.25a.75.75 0 01-1.272-.71l1.992-7.302H3.75a.75.75 0 01-.548-1.262l10.5-11.25a.75.75 0 01.913-.143z"
										clipRule="evenodd"
									/>
								</svg>
								{message.moleculeData.name}
							</h3>
							<p className="text-sm text-slate-300 mt-2 leading-relaxed">{message.moleculeData.info}</p>
						</div>

						{/* 3D Viewer */}
						{message.moleculeData.structure && (
							<div className="mt-4 rounded-xl overflow-hidden border border-white/10 bg-black/20 shadow-inner">
								<div className="bg-white/5 px-4 py-2 border-b border-white/5 flex justify-between items-center">
									<p className="text-xs text-slate-400 uppercase font-bold tracking-wider flex items-center gap-2">
										<svg
											xmlns="http://www.w3.org/2000/svg"
											viewBox="0 0 24 24"
											fill="currentColor"
											className="w-4 h-4"
										>
											<path d="M12.378 1.602a.75.75 0 00-.756 0L3 6.632l9 5.25 9-5.25-8.622-5.03zM21.75 7.93l-9 5.25v9l8.628-5.032a.75.75 0 00.372-.648V7.93zM11.25 22.18v-9l-9-5.25v8.57a.75.75 0 00.372.648l8.628 5.033z" />
										</svg>
										3D Structure
									</p>
									<span className="text-[10px] bg-blue-500/20 text-blue-300 px-2 py-0.5 rounded-full border border-blue-500/30">
										Interactive
									</span>
								</div>
								<div className="p-1">
									<MoleculeViewer
										sdf={message.moleculeData.format === "sdf" ? message.moleculeData.structure : undefined}
										pdb={message.moleculeData.format === "pdb" ? message.moleculeData.structure : undefined}
										format={message.moleculeData.format}
									/>
								</div>
							</div>
						)}
					</div>
				)}
			</div>
		</div>
	);
}
