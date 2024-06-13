/*
 */
package main

import (
	"PSnal2/socialNetwork"
	"flag"
	"fmt"
	"slices"
	"strings"
	"sync"
	"time"
)

var wg sync.WaitGroup
var lockBook sync.RWMutex

var word2id = make(map[string][]int)

// Pripravimo high in low priority kanale
const bufferSize = 100

var priorityOne = make(chan socialNetwork.Task, bufferSize)
var priorityTwo = make(chan socialNetwork.Task, bufferSize)

var semaphore chan int

// var currentIndexID = 1

func nonalphanumToWhitespace(str string) string {
	chr := "abcdefghijklmnoprstuvzxyzwq0123456789"
	chr += strings.ToUpper(chr)
	// fmt.Println(chr)

	return strings.Map(func(r rune) rune {
		if strings.ContainsRune(chr, r) {
			return r
		}
		return ' '
	}, str)
}

func index(givenTask socialNetwork.Task) {
	defer wg.Done()

	fmt.Println("Index happening:")

	currId := int(givenTask.Id)
	fmt.Println(currId)
	// currId := currentIndexID
	// fmt.Println(currId)
	// currentIndexID++

	currText := givenTask.Data

	wholeString := nonalphanumToWhitespace(currText)

	// fmt.Println(wholeString)
	wholeString = strings.ToLower(wholeString)
	// fmt.Println(wholeString)

	allWords := strings.Fields(wholeString)

	slices.Sort(allWords)
	slices.Compact(allWords)
	// fmt.Println(allWords)

	// fmt.Println(len(word2id))

	lockBook.Lock()
	for i := 0; i < len(allWords); i++ {
		if len(allWords[i]) >= 4 {
			if !slices.Contains(word2id[allWords[i]], currId) {
				word2id[allWords[i]] = append(word2id[allWords[i]], currId)
			}
		}
	}
	lockBook.Unlock()
	// fmt.Println(len(word2id))
	// fmt.Println(word2id)

	fmt.Println(givenTask.Data)

	<-semaphore

}

func search(givenTask socialNetwork.Task) []int {
	defer wg.Done()

	if len(givenTask.Data) < 1 {
		return []int{}
	}

	// this removes the \n at the end
	searchword := givenTask.Data[0 : len(givenTask.Data)-1]
	fmt.Println("Search happening:")
	fmt.Println(searchword)
	// fmt.Println(len(searchword))
	// fmt.Println(givenTask.Id)
	// fmt.Println(givenTask.Data[len(givenTask.Data)-1])
	// fmt.Println(givenTask.Data[len(givenTask.Data)-2])

	lockBook.RLock()
	ids := word2id[searchword]
	lockBook.RUnlock()

	fmt.Println(ids)

	<-semaphore

	return ids
}

func main() {

	nPtr := flag.Int("n", 20, "# of workers")
	rPtr := flag.Int("r", 1000, "rate (# tasks per second)")
	tPtr := flag.Int("t", 5, "seconds the producer will work")
	lPtr := flag.Float64("l", 0.75, "low priority (search) probability")
	flag.Parse()

	// *rPtr zahtevkov na sekundo
	var rateLimit time.Duration
	if *rPtr != 0 {
		// rateLimit = time.Duration(int(1000 / *rPtr)) * time.Millisecond
		rateLimit = time.Duration(int(1e9 / *rPtr)) * time.Nanosecond
	}

	// the producer will make requests for *tPtr seconds
	var durationTime = *tPtr * 1000

	semaphore = make(chan int, *nPtr)

	startTime := time.Now()

	// This part is the producer. The produce is also stopped on the bottom part of the code.
	var producer socialNetwork.Q
	producer.New(*lPtr)

	go func() {
		// fmt.Println("Waiting for tasks.")
		for {
			// fmt.Println("Task received 1")
			currTask := <-producer.TaskChan
			// socialNetwork.Task
			// fmt.Println(currTask.TaskType)
			if currTask.TaskType == "index" {
				priorityOne <- currTask
			} else if currTask.TaskType == "search" {
				priorityTwo <- currTask
			} else {
				fmt.Println("Unrecognised TaskType:")
				fmt.Println(currTask.TaskType)
			}

		}
	}()

	go producer.Run()

	// rate limiter:

	// Kanal omejitve prepustnosti
	rateLimiter := make(chan time.Time)

	if *rPtr != 0 {
		go func() {
			ticker := time.NewTicker(rateLimit)
			for t := range ticker.C {
				rateLimiter <- t
			}
		}()
	}

	for {

		// fmt.Println("Waiting for rate tick.")
		if *rPtr != 0 {
			<-rateLimiter
		}
		// fmt.Println("Got rate tick.")

		currentTime := time.Now()
		elapsed := currentTime.Sub(startTime)
		if elapsed.Milliseconds() > int64(durationTime) {
			// fmt.Println("About to break 1.")
			wg.Wait()
			go producer.Stop()
			// fmt.Println("About to break 2.")
			break
		}

		if len(priorityOne) > 0 {
			semaphore <- 0
			wg.Add(1)
			go index(<-priorityOne)
		} else if len(priorityTwo) > 0 {
			semaphore <- 0
			wg.Add(1)
			go search(<-priorityTwo)
		}

	}

	// the end of the producer

	elapsed := time.Since(startTime)
	fmt.Printf("Spam rate: %f MReqs/s\n", float64(producer.N[socialNetwork.LowPriority]+producer.N[socialNetwork.HighPriority])/float64(elapsed.Seconds())/1000000.0)

}
