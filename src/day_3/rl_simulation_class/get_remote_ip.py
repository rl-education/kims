import requests

print(f"The Instance IP is: {requests.get('https://api.ipify.org').text}")
