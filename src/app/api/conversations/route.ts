import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

const DATA_DIR = path.join(process.cwd(), "data");
const CONVERSATIONS_FILE = path.join(DATA_DIR, "conversations.json");

// Ensure data directory exists
if (!fs.existsSync(DATA_DIR)) {
	fs.mkdirSync(DATA_DIR, { recursive: true });
}

// Helper to read conversations
const getConversations = () => {
	if (!fs.existsSync(CONVERSATIONS_FILE)) {
		return [];
	}
	try {
		const data = fs.readFileSync(CONVERSATIONS_FILE, "utf-8");
		return JSON.parse(data);
	} catch (error) {
		console.error("Error reading conversations:", error);
		return [];
	}
};

// Helper to save conversations
const saveConversations = (conversations: any[]) => {
	try {
		fs.writeFileSync(CONVERSATIONS_FILE, JSON.stringify(conversations, null, 4));
		return true;
	} catch (error) {
		console.error("Error saving conversations:", error);
		return false;
	}
};

export async function GET() {
	const conversations = getConversations();
	return NextResponse.json(conversations);
}

export async function POST(request: Request) {
	try {
		const body = await request.json();
		const { id, messages, title } = body;

		if (!id || !messages) {
			return NextResponse.json({ error: "Missing required fields" }, { status: 400 });
		}

		const conversations = getConversations();
		const existingIndex = conversations.findIndex((c: any) => c.id === id);

		const updatedConversation = {
			id,
			title:
				title ||
				(messages.length > 0
					? messages[0].content.slice(0, 30) + (messages[0].content.length > 30 ? "..." : "")
					: "New Conversation"),
			messages,
			updatedAt: new Date().toISOString(),
		};

		if (existingIndex >= 0) {
			conversations[existingIndex] = { ...conversations[existingIndex], ...updatedConversation };
		} else {
			conversations.push({
				...updatedConversation,
				createdAt: new Date().toISOString(),
			});
		}

		saveConversations(conversations);
		return NextResponse.json({ success: true, conversation: updatedConversation });
	} catch (error) {
		return NextResponse.json({ error: "Internal Server Error" }, { status: 500 });
	}
}
