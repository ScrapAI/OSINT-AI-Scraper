from sentence_transformers import SentenceTransformer
import faiss
import os
import shutil
from typing import List, Dict


class Rag:
    def __init__(self, index_path='faiss_index'):
        self.index_path = index_path
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Modèle SentenceTransformers
        self.index = self._load_or_create_index()

    def _load_or_create_index(self):
        """Charge un index existant ou en crée un nouveau."""
        if os.path.exists(self.index_path) and os.path.isfile(self.index_path):
            index = faiss.read_index(self.index_path)
        else:
            # Création d'un nouvel index FAISS
            index = faiss.IndexFlatL2(384)  # Dimension de all-MiniLM-L6-v2
        return index

    def _save_index(self):
        """Sauvegarde l'index FAISS sur le disque."""
        faiss.write_index(self.index, self.index_path)

    def add(self, docs: Dict[str, List[str]]):
        """Ajoute des documents à la base de connaissances."""
        embeddings = []
        metadata = []
        
        for doc_id, (title, link, content) in docs.items():
            # Découpage en chunks
            chunks = self._split_text(content)
            for i, chunk in enumerate(chunks):
                embedding = self.embedding_model.encode(chunk)
                embeddings.append(embedding)
                metadata.append({
                    "doc_id": f"{doc_id}_{i}",
                    "title": title,
                    "link": link,
                    "content": chunk
                })

        embeddings = faiss.numpy.array(embeddings).astype("float32")
        self.index.add(embeddings)  # Ajout à FAISS

        # Sauvegarder les métadonnées
        self._save_metadata(metadata)

        # Sauvegarder l'index FAISS
        self._save_index()

    def _save_metadata(self, metadata):
        """Sauvegarde les métadonnées des documents."""
        with open(f"{self.index_path}_metadata.json", "w", encoding="utf-8") as f:
            import json
            json.dump(metadata, f, indent=4)

    def search(self, query: str, k: int = 3) -> str:
        """Recherche les documents pertinents."""
        embedding = self.embedding_model.encode(query).astype("float32").reshape(1, -1)
        distances, indices = self.index.search(embedding, k)

        # Charger les métadonnées
        with open(f"{self.index_path}_metadata.json", "r", encoding="utf-8") as f:
            import json
            metadata = json.load(f)

        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < 0:  # Index invalide
                continue
            meta = metadata[idx]
            relevance = max(0, min(100, (1 - distance) * 100))  # Score en pourcentage
            result = (
                f"[Source: {meta['title']}] (Pertinence: {relevance:.1f}%)\n"
                f"URL: {meta['link']}\n"
                f"{meta['content']}\n"
            )
            results.append(result)

        return "\n\n".join(results) if results else "Aucun résultat pertinent trouvé."

    def _split_text(self, text: str, chunk_size=1024, chunk_overlap=100) -> List[str]:
        """Découpe un texte en chunks avec chevauchement."""
        chunks = []
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunks.append(text[i:i + chunk_size])
        return chunks

    def clear(self):
        """Nettoie la base de connaissances."""
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(f"{self.index_path}_metadata.json"):
            os.remove(f"{self.index_path}_metadata.json")
        self.index = faiss.IndexFlatL2(384)  # Réinitialiser l'index

    def __len__(self):
        """Retourne le nombre de documents dans la base."""
        return self.index.ntotal if self.index else 0


if __name__ == "__main__":
    rag = Rag()

    # Exemple de données
    docs = {
        "1": ["La DeLorean et ses caractéristiques", "http://example.com/1", """La DeLorean présente aux personnages du film deux difficultés essentielles, qui seront un enjeu de taille tour à tour dans l'épisode I puis dans l'épisode III. En effet, le voyage dans le temps s'effectue seulement si deux conditions sine qua non sont remplies : le « convecteur temporel » doit être rechargé en énergie ; la voiture doit atteindre la vitesse de 88 miles par heure (141,619 28 km/h). Pour être précis, c'est le convecteur temporel (« Flux Capacitor » en VO) qui a besoin d'être déplacé dans l'espace à cette vitesse."""],
        "2": ["Tour Eiffel", "http://example.com/2", "La tour Eiffel mesure 324m de haut et est située à Paris."]
    }

    # Test d'ajout de documents
    rag.add(docs)
    print(f"Nombre de documents dans la base: {len(rag)}")

    # Test de recherche
    query = "qu'elle vitesse la DeLorean doit atteindre ?"
    print("\nQuestion :", query)
    print("\nRéponse :")
    print(rag.search(query))