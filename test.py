from bs4 import BeautifulSoup
import requests

url = "https://www.apartments.com/1921-owl-ct-cherry-hill-nj/thv0x2b/"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Find all images with class "item paidImageSmall"
images = soup.find_all('img', class_='item paidImageSmall')

# Extract image URLs
image_urls = [img['src'] for img in images]

# Print the results
for url in image_urls:
    print(url)
