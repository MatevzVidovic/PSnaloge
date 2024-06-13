package main

import (
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/laspp/PS-2023/vaje/naloga-1/koda/xkcd"
)

var wg sync.WaitGroup

var combiningLock sync.Mutex

type stringIntPair struct {
	key string
	val int
}

func roundRobinDictionarizeMultipleComics(comics []xkcd.Comic, startIx int, step int, goalMap map[string]int) {
	defer wg.Done()

	wordcountMap := make(map[string]int)

	for i := startIx; i < len(comics); i += step {

		comic := comics[i]

		// comic, err := xkcd.FetchComic(i)
		// if err != nil {
		// 	fmt.Print("Failed fetch!")
		// 	fmt.Print(i)
		// 	return
		// }

		title := comic.Title
		transcript := comic.Transcript
		tooltip := comic.Tooltip

		var wholeString string

		if len(transcript) > 0 {
			wholeString = title + " " + transcript
		} else {
			wholeString = title + " " + tooltip
		}

		wholeString = nonalphanumToWhitespace(wholeString)

		// fmt.Println(wholeString)
		wholeString = strings.ToLower(wholeString)
		// fmt.Println(wholeString)

		allWords := strings.Fields(wholeString)
		// fmt.Println(allWords)

		for i := 0; i < len(allWords); i++ {
			wordcountMap[allWords[i]]++
		}
	}

	// fmt.Println(wordcountMap)

	// fmt.Println("tu")

	combiningLock.Lock()
	for key, val := range wordcountMap {
		goalMap[key] += val
	}
	combiningLock.Unlock()

}

func nonalphanumToWhitespace(str string) string {
	chr := "abcdefghijklmnoprstuvzxyzwq0123456789"
	chr += strings.ToUpper(chr)
	// fmt.Println(chr)

	return strings.Map(func(r rune) rune {
		if strings.IndexRune(chr, r) >= 0 {
			return r
		}
		return ' '
	}, str)
}

func comicFetchAndSet(comicIx int, goalComics []xkcd.Comic) {
	defer wg.Done()

	comic, err := xkcd.FetchComic(comicIx)
	if err == nil {
		goalComics[comicIx-1] = comic
	} else {
		fmt.Print("Bad comic")
		fmt.Print(comicIx)
	}
}

func main() {

	comic, err := xkcd.FetchComic(0)
	if err != nil {
		fmt.Print("Failed fetch!")
		fmt.Print("Main.")
		return
	}
	// fmt.Println(comic.Id)
	var numOfComics int = comic.Id
	// var numOfComics int = 100

	comics := make([]xkcd.Comic, numOfComics)
	// fmt.Println(len(comics))

	wg.Add((numOfComics))
	for i := 1; i <= numOfComics; i++ {
		go comicFetchAndSet(i, comics)

		// comic, err := xkcd.FetchComic(i)
		// if err == nil {
		// 	goalComics[i-1] = comic
		// }
	}
	wg.Wait()

	startTime := time.Now()
	// fmt.Println(comics[1])

	// dictionarizeComic(comics[1])

	finalMap := make(map[string]int)
	numOfWorkers, _ := strconv.Atoi(os.Args[1])
	// fmt.Println("tu1")
	wg.Add(numOfWorkers)
	for i := 0; i < numOfWorkers; i++ {
		go roundRobinDictionarizeMultipleComics(comics, i, numOfWorkers, finalMap)
	}
	wg.Wait()
	// fmt.Println("tu2")

	// fmt.Print(finalMap)

	keyValPairs := make([]stringIntPair, 0)
	for key, val := range finalMap {
		keyValPairs = append(keyValPairs, stringIntPair{key, val})
	}

	// sorts them in descending order
	// fmt.Println("tu3")
	sort.SliceStable(keyValPairs, func(i, j int) bool {
		return keyValPairs[i].val > keyValPairs[j].val
	})
	// fmt.Println("tu4")

	best15slice := keyValPairs[0:15]

	fmt.Println(best15slice)

	endTime := time.Now()
	elapsed := endTime.Sub(startTime)
	fmt.Println(elapsed.Milliseconds())

}
