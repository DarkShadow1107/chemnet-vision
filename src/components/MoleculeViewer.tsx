"use client";

import { useEffect, useRef } from "react";
import * as $3Dmol from "3dmol";

interface MoleculeViewerProps {
	pdb?: string;
	sdf?: string;
	format?: string;
}

export default function MoleculeViewer({ pdb, sdf, format = "pdb" }: MoleculeViewerProps) {
	const viewerRef = useRef<HTMLDivElement>(null);

	useEffect(() => {
		if (!viewerRef.current) return;

		const element = viewerRef.current;
		const config = { backgroundColor: "#1f2937" }; // gray-800
		const viewer = $3Dmol.createViewer(element, config);

		if (pdb) {
			viewer.addModel(pdb, "pdb");
		} else if (sdf) {
			viewer.addModel(sdf, "sdf");
		}

		viewer.setStyle({}, { stick: {} });
		viewer.zoomTo();
		viewer.render();

		return () => {
			// Cleanup if necessary, though 3Dmol doesn't have a strict destroy method exposed easily
			viewer.clear();
		};
	}, [pdb, sdf, format]);

	return <div ref={viewerRef} className="w-full h-64 rounded-lg overflow-hidden border border-gray-600 relative" />;
}
