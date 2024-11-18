import re

def validate_age(text):
   try:
       age = int(text)
       return 0 <= age <= 120
   except:
       return False

def validate_gender(text):
   return str(text).lower() in ['m', 'f', 'male', 'female']

def validate_name(text):
   return bool(re.match(r'^[A-Za-z\s\-\.]{2,30}$', str(text)))

def validate_phone(text):
   cleaned = re.sub(r'[^\d]', '', str(text))
   return len(cleaned) == 10 and bool(re.match(r'\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4}', str(text)))

def validate_emergency_phone(text):
   cleaned = re.sub(r'[^\d]', '', str(text))
   return len(cleaned) == 10 and bool(re.match(r'^(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}$', str(text)))
