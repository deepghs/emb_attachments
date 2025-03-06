import mimetypes

try:
    import pillow_avif
except (ImportError, ModuleNotFoundError) as err:
    pass

mimetypes.add_type('image/webp', '.webp')

MARK = object()
