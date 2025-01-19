from seleniumbase import SB  # Import the SeleniumBase Manager
from urllib.parse import urlencode, urlparse, parse_qs, urlunparse

class GoogleSerper:
    def __init__(self):
        self.position_index = 1
        self.current_page = 1
    def increment_page(self):
        self.current_page += 1
    def update_url_param(self, url, param_name, new_value):
        # Parse the URL
        parsed_url = urlparse(url)
        # Parse query parameters into a dictionary
        query_params = parse_qs(parsed_url.query)
        
        # Update the parameter value
        query_params[param_name] = [new_value]
        
        # Encode back the query string
        updated_query = urlencode(query_params, doseq=True)
        
        # Return the new URL
        return urlunparse(parsed_url._replace(query=updated_query))
    def serp_organic_results(self, query: str, page_num: int) -> dict | None:
            if page_num < 1:
                raise ValueError("Page number should be greater or equal to 1")
            with SB(
                undetectable=True,
                headless2=True,
            ) as sb:  # Initialize SeleniumBase Manager
                with open("src/google_serper.js", "r") as GS_file:
                    GS_script = GS_file.read()
                    sb.open("https://google.com")

                    # Accept cookies if present
                    sb.js_click_if_present("#W0wltc")
                    
                    # Type the query in the search box
                    sb.type('//*[@id="APjFqb"]', query+"\n")
                    
                    sb.wait_for_ready_state_complete()

                    # Update the page number (if different to 1) once the page is loaded and thus when the url is formed
                    if page_num != 1:
                        current_url = sb.get_current_url()
                        # Update the start parameter in the URL to navigate to the wanted page
                        updated_url = self.update_url_param(current_url, "start", str((page_num-1)*10))
                        sb.open(updated_url)
                        sb.wait_for_ready_state_complete()

                    organic_results, self.position_index = sb.execute_script(GS_script, self.position_index)
                    return organic_results
    
# Run the function
if __name__ == "__main__":
    import json
    # Example usage
    # Create an instance of the GoogleSE class
    google = GoogleSerper()

    # Get the first page of organic results for the query "youtube"
    print(json.dumps(google.serp_organic_results(query='news', page_num=1), sort_keys=True, indent=4))

    # Get the second page of organic results for the query "youtube"
    # Note: We need to notify the GoogleSE instance that we are navigating to the next page so that the position index gets updated to its right value
    google.increment_page()
    print(json.dumps(google.serp_organic_results(query='news', page_num=2), sort_keys=True, indent=4))
