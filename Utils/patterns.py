PATTERNS = {
   'age': {
       'regex': r'^\d{1,2}$',
       'invalid_patterns': [
           r'.*Guest.*',
           r'.*roommate.*',
           r'.*\*.*',
           r'.*[A-Za-z]+.*',
           r'^0[0-9].*$',
           r'^(1[2-9]\d|[2-9]\d{2}|[1-9]\d{3,})$'
       ]
   },
   'gender': {
       'regex': r'^[MF]$|^(male|female)$',
       'invalid_patterns': [
           r'.*Guest.*',
           r'.*roommate.*',
           r'.*\*.*',
           r'^\d+$'
       ]
   },
   'name': {
       'regex': r'^[A-Za-z\s\-\.]{2,30}$',
       'invalid_patterns': [
           r'^\d+$',
           r'^Guest\s*Name$',
           r'\(\d{3}\)\s*\d{3}-\d{4}',
           r'.*Report$',
           r'.*Expedition$',
           r'.*Insurance.*',
           r'.*Adventures.*',
           r'.{31,}'
       ]
   },
   'phone': {
       'regex': r'\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4}',
       'invalid_patterns': [
           r'^[A-Za-z\s\-\.]+$',
           r'^Guest\s*Name$',
           r'.*Report$',
           r'.*Expedition$',
           r'.*Insurance.*',
           r'.*Adventures.*'
       ]
   },
   'emergency_phone': {
       'regex': r'^(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}$',
       'invalid_patterns': [
           r'.*Guest.*',
           r'.*roommate.*',
           r'.*\*.*',
           r'.*[A-Za-z]+.*'
       ]
   }
}