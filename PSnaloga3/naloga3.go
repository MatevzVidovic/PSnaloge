package main

import (
	"flag"
	"fmt"
	"math/rand"
	"net"
	"reflect"
	"strconv"
	"time"
)

type message struct {
	data   []byte
	length int
}

var start chan bool
var stopHeartbeat bool
var N int
var id int

func checkError(err error) {
	if err != nil {
		panic(err)
	}
}

func receive(conn *net.UDPConn, numOfSecs int) (message, error) {
	// Poslušamo

	conn.SetDeadline(time.Now().Add(time.Duration(numOfSecs) * time.Second))

	fmt.Println("Agent", id, "listening")
	buffer := make([]byte, 1024)
	// Preberemo sporočilo
	mLen, err := conn.Read(buffer)

	// checkError(err)

	// fmt.Println("Agent", id, "Received:", string(buffer[:mLen]))
	// // Vrnemo sporočilo
	// rMsg := message{}
	// rMsg.data = append(rMsg.data, buffer[:mLen]...)
	// rMsg.length = mLen
	// return rMsg, err

	if err == nil {
		fmt.Println("Agent", id, "Received:", string(buffer[:mLen]))
		// Vrnemo sporočilo
		rMsg := message{}
		rMsg.data = append(rMsg.data, buffer[:mLen]...)
		rMsg.length = mLen
		return rMsg, err
	} else {
		return message{}, err
	}
	// checkError(err)
}

func receiveZeroth(conn *net.UDPConn, numOfSecs int) bool {
	// Poslušamo

	conn.SetDeadline(time.Now().Add(time.Duration(numOfSecs) * time.Second))

	fmt.Println("Agent", id, "listening")
	buffer := make([]byte, 1024)
	// Preberemo sporočilo
	mLen, err := conn.Read(buffer)

	rMsg := message{}

	if err != nil {
		return false
	}

	checkError(err)
	fmt.Println("Agent", id, "Received:", string(buffer[:mLen]))
	// Vrnemo sporočilo
	rMsg.data = append(rMsg.data, buffer[:mLen]...)
	rMsg.length = mLen
	return true
}

func send(addr *net.UDPAddr, msg message, numOfSecs int) {
	// Odpremo povezavo
	conn, err := net.DialUDP("udp", nil, addr)
	checkError(err)
	defer conn.Close()

	// conn.SetDeadline(time.Now().Add(time.Duration(numOfSecs) * time.Second))
	// Pripravimo sporočilo
	// sMsg := fmt.Sprint(id) + "-"
	// sMsg = string(msg.data[:msg.length]) + sMsg

	sMsg := string(msg.data[:msg.length])
	_, err = conn.Write([]byte(sMsg))
	checkError(err)
	// fmt.Println("Agent", id, "sent", sMsg, "to", addr)
	// Ustavimo heartbeat servis
	// stopHeartbeat = true
}

func remove(s []int, ix int) []int {
	s[ix] = s[len(s)-1]
	return s[:len(s)-1]
}

func sampleIds(sampleSize int, maxIdExclusive int, myId int) []int {
	// This will give a random sample of ids. Preventing myId from coming into the sample.
	chosenInts := make([]int, 0, sampleSize)

	availableInts := make([]int, 0)
	for i := 0; i < maxIdExclusive; i++ {
		availableInts = append(availableInts, i)
	}

	availableInts = remove(availableInts, myId)

	// fmt.Println(maxIdExclusive)

	for len(chosenInts) < sampleSize {
		newIx := rand.Intn(len(availableInts))
		// fmt.Println(newInt)
		chosenInts = append(chosenInts, availableInts[newIx])
		availableInts = remove(availableInts, newIx)
		// fmt.Println(availableInts)
	}

	return chosenInts
}

func sendToK(addresses []*net.UDPAddr, msg message, K int, myId int, numOfSecs int) {

	chosenIxs := sampleIds(K, len(addresses), myId)

	// fmt.Print(chosenIxs)

	for ix := 0; ix < len(chosenIxs); ix++ {
		currAddr := addresses[chosenIxs[ix]]
		// print(currAddr)
		send(currAddr, msg, numOfSecs)
	}

	sMsg := string(msg.data[:msg.length])

	strOfChosenIxs := ""

	for ix := 0; ix < len(chosenIxs); ix++ {
		strOfChosenIxs += fmt.Sprint(chosenIxs[ix]) + ", "
	}

	fmt.Println("Agent", id, "sent", sMsg, "to", strOfChosenIxs)

}

func hasAlreadyBeenReceived(newMsg message, msgsSoFar []message) bool {

	for i := 0; i < len(msgsSoFar); i++ {
		if reflect.DeepEqual(newMsg, msgsSoFar[i]) {
			return true
		}
	}

	return false
}

func main() {

	// identifikator procesa id: celo število, ki identificira posamezen proces znotraj skupine,
	// število vseh procesov v skupini N,
	// število sporočil M, ki jih bo razširil glavni proces,
	// število sporočil K, ki jih bo vsak proces ob prvem prejemu posredoval ostalim. S tem parametrom nastavljamo tudi metodo razširjanja sporočil. Če nastavimo K==N-1, naj se uporabi nestrpno razširjanje. Za K<N-1 pa naj se uporabi razširjanje z govoricami.

	// Preberi argumente
	idPtr := flag.Int("id", 0, "# process id")
	NPtr := flag.Int("N", 2, "total number of processes")
	MPtr := flag.Int("M", 2, "total number of messages")
	KPtr := flag.Int("K", 1, "how many to send to")
	portPtr := flag.Int("p", 15000, "# start port")
	flag.Parse()

	// is static
	id = *idPtr

	zerothPort := *portPtr
	basePort := *portPtr + id
	N := *NPtr
	M := *MPtr
	K := *KPtr

	msgsSoFar := make([]message, 0)
	// Ustvari potrebne mrežne naslove
	// rootAddr, err := net.ResolveUDPAddr("udp", fmt.Sprintf("localhost:%d", rootPort))
	// checkError(err)

	localAddr, err := net.ResolveUDPAddr("udp", fmt.Sprintf("localhost:%d", basePort))
	checkError(err)

	localConn, err := net.ListenUDP("udp", localAddr)
	checkError(err)

	allProcPorts := make([]*net.UDPAddr, 0)

	for i := 0; i < N; i++ {

		if i == id {
			allProcPorts = append(allProcPorts, nil)
			continue
		}

		currPort := zerothPort + i
		remoteAddr, err := net.ResolveUDPAddr("udp", fmt.Sprintf("localhost:%d", currPort))
		checkError(err)

		allProcPorts = append(allProcPorts, remoteAddr)
	}

	// Ustvari kanal, ki bo signaliziral, da so vsi procesi pripravljeni
	// start = make(chan bool)

	// Zaženemo heartbeat servis, ki čaka, na javljanje vseh udeleženih procesov
	// stopHeartbeat = false
	// go heartBeat(rootAddr)

	// Izmenjava sporočil
	if id == 0 {

		time.Sleep(2 * time.Second)

		go func() {
			for receiveZeroth(localConn, 10) {
			}
			localConn.Close()
		}()

		for i := 0; i < M; i++ {

			currData := []byte(strconv.Itoa(i))
			rMsg := message{}
			rMsg.data = append(rMsg.data, currData...)
			rMsg.length = len(currData)
			sendToK(allProcPorts, rMsg, K, id, 100)
			time.Sleep(100 * time.Millisecond)

			// fmt.Println(string(rMsg.data[:rMsg.length]) + "0")
		}

	} else {

		for {
			rMsg, err := receive(localConn, 10)

			if err != nil {
				break
			}

			if !hasAlreadyBeenReceived(rMsg, msgsSoFar) {
				sendToK(allProcPorts, rMsg, K, id, 100)
			}
			msgsSoFar = append(msgsSoFar, rMsg)
		}

		localConn.Close()

	}

}
