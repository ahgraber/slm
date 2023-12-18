import logging

# add nullhandler to prevent a default configuration being used if the calling application doesn't set one
logging.getLogger(__name__).addHandler(logging.NullHandler())
