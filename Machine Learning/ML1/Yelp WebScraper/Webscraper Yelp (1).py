#!/usr/bin/env python
# coding: utf-8

# In[3]:


from bs4 import BeautifulSoup
import requests

# Sample HTML provided for star rating and review count
star_rating_html = """
<div class="info-box__09f24__UA82F css-1qn0b6x">
    <!-- ... (other HTML elements) ... -->
    <div class="arrange-unit-fill__09f24__CUubG css-v3nuob">
        <span class=" css-gutk1c" data-font-weight="semibold">4.4<!-- --> </span>
        <span class="review-count__09f24__FPtB_ css-chan6m">(3,497 reviews)</span>
    </div>
    <!-- ... (other HTML elements) ... -->
</div>
"""

# Parse the star rating and review count from the sample HTML
star_soup = BeautifulSoup(star_rating_html, 'html.parser')
star_rating_element = star_soup.find('span', class_='css-gutk1c')
star_rating = star_rating_element.text.strip() if star_rating_element else 'Not Found'
review_count_element = star_soup.find('span', class_='review-count__09f24__FPtB_')
review_count = review_count_element.text.strip() if review_count_element else 'Not Found'

# URL of the Yelp page you want to scrape
url = "https://www.yelp.ca/biz/pai-northern-thai-kitchen-toronto-5?osq=Restaurants"

# Send an HTTP GET request to the URL
response = requests.get(url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the HTML content of the Yelp page
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract restaurant price range
    price_range_element = soup.find('span', class_='priceRange__09f24__epbmu')
    price_range = price_range_element.text.strip() if price_range_element else 'Not Found'

    category_element = soup.find('span', class_='css-qgunke')
    category = category_element.text.strip() if category_element else 'Not Found'

    # Extract restaurant name (using a more general approach)
    restaurant_name_element = soup.find('h1')
    restaurant_name = restaurant_name_element.text.strip() if restaurant_name_element else 'Not Found'

    # Extract restaurant rating (using a more general approach)
    restaurant_rating_element = soup.find('span', class_='css-1d8srnw')
    restaurant_rating = restaurant_rating_element.text.strip() if restaurant_rating_element else 'Not Found'

    # Extract additional info from the provided HTML snippet on the Yelp page
    info_box_element = soup.find('div', class_='info-box__09f24__UA82F')
    if info_box_element:
        price_range_element = info_box_element.find('span', class_='priceRange__09f24__epbmu')
        price_range = price_range_element.text.strip() if price_range_element else 'Not Found'
        
        category_element = info_box_element.find('span', class_='css-qgunke')
        category = category_element.text.strip() if category_element else 'Not Found'

    # Print the extracted restaurant information
    print(f'Restaurant Name: {restaurant_name}')
    print(f'Restaurant Rating: {restaurant_rating}')
    print(f'Star Rating: {star_rating}')
    print(f'Review Count: {review_count}')
    print(f'Price Range: {price_range}')
    print(f'Category: {category}')
else:
    print(f'Failed to retrieve the page. Status code: {response.status_code}')

# Sample HTML provided for the specific reviewer
html = """
<div class=" css-na7xnn"><div class=" css-1qn0b6x" role="region" aria-label="Jamie S.">
    <!-- ... (previous code) ... -->
</div></div>
<div class="review-content__09f24__K12aw review-content-unhidden-photos__09f24__P27AR css-1fyi5wc">
    <p class="review-text__09f24__f_0gn css-qgunke">You can't tell from the outside but this place is MASSIVE 
    ..…<span class=" css-16sa6ah"> read more</span></p>
    <div class="review-unhidden-photos-grid__09f24__S1vhb css-1qn0b6x">
        <!-- ... (photo links) ... -->
    </div>
</div>
"""

# Parse the HTML content
soup = BeautifulSoup(html, 'html.parser')

# Extract the reviewer's name
reviewer_name_element = soup.find('a', class_='css-19v1rkv')
reviewer_name = reviewer_name_element.text.strip() if reviewer_name_element else 'Not Found'

# Extract the reviewer's photo URL
reviewer_photo_element = soup.find('img', class_='css-1pz4y59')
reviewer_photo_url = reviewer_photo_element['src'] if reviewer_photo_element else 'Not Found'

# Extract the reviewer's elite badge (if available)
elite_badge_element = soup.find('span', class_='css-1adhs7a')
elite_badge = elite_badge_element.text.strip() if elite_badge_element else 'Not Elite'

# Extract the reviewer's text review
text_review_element = soup.find('p', class_='review-text__09f24__f_0gn')


# Sample HTML provided for the specific reviewer
html = """
<div class=" css-na7xnn"><div class=" css-1qn0b6x" role="region" aria-label="Jamie S.">
    <!-- ... (previous code) ... -->
</div></div>
<div class="review-content__09f24__K12aw review-content-unhidden-photos__09f24__P27AR css-1fyi5wc">
    <p class="review-text__09f24__f_0gn css-qgunke">You can't tell from the outside but this place is MASSIVE 
    ..…<span class=" css-16sa6ah"> read more</span></p>
    <div class="review-unhidden-photos-grid__09f24__S1vhb css-1qn0b6x">
        <!-- ... (photo links) ... -->
    </div>
</div>
"""

# Parse the HTML content
soup = BeautifulSoup(html, 'html.parser')

# Extract the reviewer's name
reviewer_name_element = soup.find('a', class_='css-19v1rkv')
reviewer_name = reviewer_name_element.text.strip() if reviewer_name_element else 'Not Found'

# Extract the reviewer's photo URL
reviewer_photo_element = soup.find('img', class_='css-1pz4y59')
reviewer_photo_url = reviewer_photo_element['src'] if reviewer_photo_element else 'Not Found'

# Extract the reviewer's elite badge (if available)
elite_badge_element = soup.find('span', class_='css-1adhs7a')
elite_badge = elite_badge_element.text.strip() if elite_badge_element else 'Not Elite'

# Extract the reviewer's text review
text_review_element = soup.find('p', class_='review-text__09f24__f_0gn')



# Sample HTML provided for the specific reviewer
html = """
<div class=" css-na7xnn"><div class=" css-1qn0b6x" role="region" aria-label="Jamie S.">
    <div class="arrange__09f24__LDfbs gutter-1__09f24__yAbCL vertical-align-middle__09f24__zU9sE css-1qn0b6x">
        <div class="arrange-unit__09f24__rqHTg css-1qn0b6x">
            <div class="css-eqfjza">
                <a href="/user_details?userid=w5MzmH3rfJLS6c-8q4kebQ" class="css-vzslx5" target="_self">
                    <img class=" css-1pz4y59" src="https://s3-media0.fl.yelpcdn.com/photo/mEuM_PLqpY0GxjdBtllRYA/30s.jpg" alt="Photo of Jamie S." height="40" width="40" loading="lazy" draggable="true">
                </a>
            </div>
        </div>
        <div class="arrange-unit__09f24__rqHTg arrange-unit-fill__09f24__CUubG css-1qn0b6x">
            <div class="user-passport-info css-1qn0b6x">
                <span class="fs-block css-ux5mu6" data-font-weight="bold">
                    <a href="/user_details?userid=w5MzmH3rfJLS6c-8q4kebQ" class="css-19v1rkv" role="link">Jamie S.</a>
                </span>
                <div class="elite-badge__09f24__dykWK css-1nr7t16">
                    <a href="/elite" class="css-14o0nng">
                        <span class="css-1adhs7a">Elite 23</span>
                    </a>
                </div>
            </div>
            <div class=" css-1jq1ouh">
                <div class="user-passport-stats__09f24__NQxB4 css-1qn0b6x">
                    <!-- ... (other user stats) ... -->
                </div>
            </div>
        </div>
    </div>
</div>
</div>
"""

# Parse the HTML content
soup = BeautifulSoup(html, 'html.parser')

# Extract the reviewer's name
reviewer_name_element = soup.find('a', class_='css-19v1rkv')
reviewer_name = reviewer_name_element.text.strip() if reviewer_name_element else 'Not Found'

# Extract the reviewer's photo URL
reviewer_photo_element = soup.find('img', class_='css-1pz4y59')
reviewer_photo_url = reviewer_photo_element['src'] if reviewer_photo_element else 'Not Found'

# Extract the reviewer's elite badge (if available)
elite_badge_element = soup.find('span', class_='css-1adhs7a')
elite_badge = elite_badge_element.text.strip() if elite_badge_element else 'Not Elite'

# You can continue extracting other information if needed

# Print the extracted information
print(f'Reviewer Name: {reviewer_name}')
print(f'Reviewer Photo URL: {reviewer_photo_url}')
print(f'Elite Badge: {elite_badge}')

# Sample HTML provided for the specific reviewer
html = """
<div class=" css-na7xnn"><div class=" css-1qn0b6x" role="region" aria-label="Jamie S.">
    <!-- ... (previous code) ... -->
</div></div>
<div class="review-content__09f24__K12aw review-content-unhidden-photos__09f24__P27AR css-1fyi5wc">
    <p class="review-text__09f24__f_0gn css-qgunke">You can't tell from the outside but this place is MASSIVE 
    ..…<span class=" css-16sa6ah"> read more</span></p>
    <div class="review-unhidden-photos-grid__09f24__S1vhb css-1qn0b6x">
        <!-- ... (photo links) ... -->
    </div>
</div>
"""

# Parse the HTML content
soup = BeautifulSoup(html, 'html.parser')

# Extract the reviewer's text review
text_review_element = soup.find('p', class_='review-text__09f24__f_0gn')
text_review = text_review_element.text.strip() if text_review_element else 'No Review Found'

# Print the extracted information
print(f'Reviewer Text Review: {text_review}')


# In[6]:


import pandas as pd  # Import the pandas library

# Rest of your code here...
# Create a dictionary with the extracted data
data = {
    'Reviewer Name': [reviewer_name],
    'Star Rating': [star_rating],
    'Review Count': [review_count],
    'Reviewer Text Review': [text_review]
}

# Create a DataFrame from the dictionary
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('yelp_review.csv', index=False)

print('CSV file has been created successfully.')


# In[ ]:




