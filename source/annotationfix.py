import copy
import json
import numpy as np
import sys

'''
The main method outputs 2 different modified annotations:
1) With images containing invalid categories removed entirely; then pruned
2) With images containing invalid categories having the annotations of those categories removed; then pruned

Example input for command line:
python annotationfix.py annotations.json output/annotations 3 True
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
            if counts[cat] >= low and counts[cat] - count < low: # Do not allow a sparse category to be removed
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

# python3 annotationfix.py <input> <output> <upper> <test?>
def main(input, output, upper, test): #This can most likely be broken up into smaller functions
    # Other constants
    path_deleted = '_deleted.json'
    path_edited = '_edited.json'

    # Get data
    imgs, cats = loadData(input)
    counts, countMap = getCounts(imgs, cats)
    low, high = getThresholds(counts, upper)

    # Print info
    print(getInfo(imgs, counts))

    # Remove categories
    remove, keep = getRemoveCats(counts, low)

    # Delete images
    imgs_deleted = removeImgsWithCats(imgs, remove)
    counts_deleted, countMap_deleted = getCounts(imgs_deleted, keep)

    # Remove annotations instead
    imgs_edited = removeCatsFromImgs(imgs, remove)
    counts_edited, countMap_edited = getCounts(imgs_edited, keep)

    # Prune both
    imgs_deleted, counts_deleted = pruneImgs(imgs_deleted, counts_deleted, countMap_deleted, low, high)
    imgs_edited, counts_edited = pruneImgs(imgs_edited, counts_edited, countMap_edited, low, high)

    # Print info
    print(getInfo(imgs_deleted, counts_deleted))
    print(getInfo(imgs_edited, counts_edited))

    # Run testing
    if test:
        runTests(imgs_deleted, keep, counts_deleted, 'deleted')
        runTests(imgs_edited, keep, counts_edited, 'edited')
    
    # Save the two jsons
    saveData(output + path_deleted, imgs_deleted, keep)
    saveData(output + path_edited, imgs_edited, keep)
    # Done
        
# Run file if applicable
if __name__ == '__main__':
    if len(sys.argv) >= 4:
        # Command line
        input = sys.argv[1]
        output = sys.argv[2]
        upper = float(sys.argv[3])
        test = len(sys.argv) >= 5 and sys.argv[4] == 'True'
        main(input, output, upper, test)
    else:
        print('Too few arguments.  Required: input output upper')
