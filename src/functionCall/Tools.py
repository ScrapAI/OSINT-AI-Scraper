
import ollama_client
from pydantic import BaseModel, Field
from typing import Optional
from src.WebpageResultsRetrieval import AIWebpageScraper
from src.WebContentExtractor import WebContentExtractor
from urllib.parse import urlparse

class ScrapingInput(BaseModel):
    url: str = Field(..., description="L'URL complète de la page web à analyser")
    extraction_description: Optional[str] = Field(None, description="Description optionnelle des informations spécifiques à extraire")

class WebTools:
    def __init__(self):
        pass

    def _is_valid_url(self, url: str) -> bool:
        """Vérifie si l'URL est valide"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False

    async def scrape_webpage(self, url: str, extraction_description: Optional[str] = None) -> str:
        """Utiliser cet outil pour extraire des informations d'une page web.
        
        Args:
            url: L'URL de la page web à analyser (doit commencer par http:// ou https://)
            extraction_description: Description optionnelle des informations spécifiques à extraire
            
        Returns:
            str: Le contenu extrait de la page web ou un message d'erreur
        """
        if not self._is_valid_url(url):
            return "URL invalide. Veuillez fournir une URL complète commençant par http:// ou https://"
        
        try:
            # scraper = AIWebpageScraper(url)
            scraper = WebContentExtractor(url=url, sb=None)
            page_source = scraper.scrape_page()
            
            if not page_source:
                return "Échec du scraping de la page."
            
            extracted_text = scraper.extract_html_content(page_source) or "Aucun contenu lisible extrait."
            
            # cleaned_content = scraper.clean_body_content(extracted_text)
            # if extraction_description:
            #     # Si une description spécifique est fournie, utiliser le parsing
            #     content_chunks = scraper.split_dom_content(cleaned_content)
            #     # return scraper.parse_with_ollama(content_chunks, extraction_description) # je trouve plus le AIWebpageScraper
            #     return self.parse_with_ollama(content_chunks, extraction_description)
            
            return extracted_text

        except Exception as e:
            return f"Erreur lors du scraping: {str(e)}"
    
    def parse_with_ollama(self, content_chunks, extraction_description):
        """Envoie les morceaux de contenu à Ollama pour analyse."""
        results = []
        for chunk in content_chunks:
            prompt = f"Extrait suivant : {chunk}\n\n{extraction_description}"
            result = ollama_client.query(prompt)  
            results.append(result)
        return "\n".join(results)

    def get_tools(self):
        """Retourne la liste des outils disponibles"""
        return [
            Tool(
                name="scrape_webpage",
                description="Extraire des informations d'une page web",
                func=self.scrape_webpage
            )
        ]