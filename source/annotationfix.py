import copy
import json
import numpy as np
import sys

'''
The main method outputs 2 different modified annotations:
1) With images containing invalid categories removed entirely; then pruned (Removed)
2) With images containing invalid categories having the annotations of those categories removed; then pruned

Example input for command line:
python annotationfix.py annotations.json output/annotations 2 0.2 True > output/output.txt
'''

# Loads a json file
def loadData(path):
    # Open file
    with open(path) as file:
        # Get and return the data as two values
        data = json.load(file)
        return data['imgs'], data['types']

# saves a json file
def saveData(path, imgs, cats):
    # Zip into big dictionary
    data = {'imgs': imgs, 'types': cats}
    # Credit to https://stackoverflow.com/a/12309296
    with open(path, 'w+', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    # Done

# Gets totals for each category and a map of images to how many of each category they contain
def getCounts(imgs, cats):
    counts = dict.fromkeys(cats, 0)
    countMap = {}
    for id, img in imgs.items():
        imgMap = {}
        for obj in img['objects']:
            cat = obj['category']
            counts[cat]+=1
            if cat in imgMap:
                imgMap[cat]+=1
            else:
                imgMap[cat]=1
        countMap[id] = imgMap
    return counts, countMap

# Gets the length of images and the counts of each of the categories as a string
def getInfo(imgs, counts):
    return ("Images: " + str(len(imgs)), "Categories: " +
        str(counts))

# Return a list of categories to remove
def getRemoveCats(counts, low):
    remove = []
    keep = []
    for (cat, count) in counts.items():
        if count < low:
            remove.append(cat)
        else:
            keep.append(cat)
        
    return remove, keep # Return both

# Completely removes images containing various categories
def removeImgsWithCats(imgs, cats):
    def imgHasCat(img):
        for obj in img['objects']:
            if obj['category'] in cats:
                return True
        return False
    return {id:img for (id, img) in imgs.items() if not imgHasCat(img)}

# Removes all categories from images and returns the new imgs dictionary
def removeCatsFromImgs(imgs, cats):
    def removeCatsFromImg(img):
        newImg = copy.deepcopy(img)
        newImg['objects'] = [obj for obj in newImg['objects'] if obj['category'] not in cats]
        return newImg
    return {id:removeCatsFromImg(img) for (id, img) in imgs.items()}

# Gets lower, upper thresholds
def getThresholds(counts, upper):
    mean = np.mean(list(counts.values()))
    return mean, upper*mean

# Greedy search to avoid np
def pruneImgs(imgs, counts, countMap, low, high):
    highCats = [cat for (cat, count) in counts.items() if count > high] # Find categories that are too high
    counts = copy.deepcopy(counts) # Copy counts
    # Sort keys in order of removable categories
    def highCatTotal(key):
        imgMap = countMap[key]
        sum = 0
        for cat, count in imgMap.items():
            if cat in highCats:
                sum += count
        return sum
    
    newImgs = {}
    for key in sorted(imgs.keys(), key=highCatTotal, reverse=True):
        imgMap = countMap[key]
        sum = 0
        for cat, count in imgMap.items():
            if counts[cat] - count < low: # Do not allow a sparse category to be removed
                sum = 0
                break
            elif cat in highCats: # In favor of removing
                sum+=count
        
        if sum == 0: # If this is NOT useful to remove, keep it
            newImgs[key] = imgs[key]
        else: # Update counts, high categories
            for cat, count in imgMap.items():
                val = counts[cat] - count
                counts[cat] = val
                if cat in highCats and val <= high:
                    highCats.remove(cat)
    return newImgs, counts # Return final list and counts associated

# Simple sanity check for now
def runTests(imgs, cats, counts, msg):
    newCounts, newCountMap = getCounts(imgs, cats)
    if newCounts == counts:
        print(msg + ' passed!')
    else:
        print(msg + ' failed!')
        print('Actual:')
        print(str(newCounts))
        print('-----------')
        print('Expected:')
        print(str(counts))

# Returns the topmost folder of a filepath
def getFolder(str):
    return str[:str.index('/')] if str.index('/') >= 0 else ''

# Removes a percentage of background images
def removeBackground(imgs, background):
    # Filter background images
    background_imgs = {}
    foreground_imgs = {}
    for id,img in imgs.items():
        if len(img['objects']) == 0:
            background_imgs[id]=img
        else:
            foreground_imgs[id]=img
    # How many background images to use
    use_bg = int(background * len(background_imgs))
    # Shuffle and slice
    background_keys = list(background_imgs.keys())
    np.random.shuffle(background_keys)
    background_keys = background_keys[:use_bg]
    # Add back
    for key in background_keys:
        foreground_imgs[key] = background_imgs[key]
    # Return the result
    return foreground_imgs

# python3 annotationfix.py <input> <output> <upper> <background> <test?>
def main(input, output, upper, background, test): #This can most likely be broken up into smaller functions
    # Other constants
    path_edited = '.json'

    # Get data
    imgs, cats = loadData(input)

    # Get splits
    folders = set([getFolder(img['path']) for (id, img) in imgs.items() if img['path'].index('/')>=0])
    print(folders)

    # Remove useless categories
    counts, countMap = getCounts(imgs, cats)
    low, high = getThresholds(counts, upper)

    # Print info
    print(getInfo(imgs, counts))

    # Remove categories
    remove, keep = getRemoveCats(counts, low)

    # Remove annotations
    imgs_edited = removeCatsFromImgs(imgs, remove)

    # Cacluate separately based on splits
    for folder in folders:
        # Filter
        folder_imgs = {id:img for (id, img) in imgs_edited.items() if getFolder(img['path'])==folder}
        folder_counts, folder_countMap = getCounts(folder_imgs, keep)

        # Prune
        folder_imgs, folder_counts = pruneImgs(folder_imgs, folder_counts, folder_countMap, low, high)

        # Print start of info
        print(folder + ":")
        # Print initial info
        print("-------")
        print(getInfo(folder_imgs, folder_counts))

        # New: Remove background images
        folder_imgs = removeBackground(folder_imgs, background)

        # Print end info
        print("-------")
        print(getInfo(folder_imgs, folder_counts))

        # Run testing
        if test:
            runTests(folder_imgs, keep, folder_counts, folder)
    
        # Save the json
        saveData(output + '_' + folder + path_edited, folder_imgs, keep)
    # Done


# Run file if applicable
if __name__ == '__main__':
    if len(sys.argv) >= 5:
        # Command line
        input = sys.argv[1]
        output = sys.argv[2]
        upper = float(sys.argv[3])
        background = float(sys.argv[4])
        test = len(sys.argv) >= 6 and sys.argv[5] == 'True'
        main(input, output, upper, background, test)
    else:
        print('Too few arguments.  Required: input output background upper')
