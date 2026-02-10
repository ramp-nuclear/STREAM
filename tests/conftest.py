from hypothesis import settings

settings.register_profile("fast", max_examples=10, deadline=None)
settings.register_profile("thorough", max_examples=500, deadline=None)
