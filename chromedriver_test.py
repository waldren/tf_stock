from selenium import webdriver
import chromedriver_binary

# chrome_options=webdriver.ChromeOptions()
# chrome_options.add_argument("--headless")
# chrome_options.add_argument("--no-sandbox")
# chrome_options.add_argument("window-size=1400,2100") 
# chrome_options.add_argument('--disable-gpu')

# driver = webdriver.Chrome()
driver = webdriver.Remote('http://selenium:4444')
driver.get("http://github.com")
driver.close()