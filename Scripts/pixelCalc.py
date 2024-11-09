from PIL import Image

#sensor dimensions
sensor_width = 0.0063
focal_length = 0.012


"""takes input of altitute of plane and width of the image, then calculates 
    and returns how many pixels will be map to one meter

    @param Altitute: altitude in meters that photo is captured from
    @param Image_w: width of the camera capture
    @return pixels_per_meter rounded to the nearest integer"""
def calc_pixel_size(altitude: int, image: Image.Image):

    global sensor_width
    global focal_length

    image_w, image_h = image.size

    resolution = (altitude * sensor_width)/(focal_length * image_w)
    pixels_per_meter = 1/resolution

    return int(pixels_per_meter)


"""takes input of pixels per meter and actual size of object, returns the size that object should be
    @param pixels_per_meter
    @param size: list [length, width] <- actual length and width of object in meters
    @return [length in pixels, width in pixels]"""
def calc_required_size(pixels_per_meter: int, size: [int, int]):
    length_pixels = size[0] * pixels_per_meter
    width_pixels = size[1] * pixels_per_meter

    return [length_pixels, width_pixels]


"""resizes image base on the size
    @param size: [new length, new width] <- in pixels
    @param image: image that needs to be resized"""
def resize(size: [int, int], image: Image.Image):
    return image.resize(size[0], size[1])

