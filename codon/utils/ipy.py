try:
    from IPython.display import FileLink, display
    HAS_IPY = True
except:
    HAS_IPY = False

if HAS_IPY:
    def show(path):
        display(FileLink(path))