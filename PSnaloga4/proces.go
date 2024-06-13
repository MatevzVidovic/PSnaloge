package main

import (
	"errors"
	"fmt"
	"sync"
)

type Todo struct {
	Task      string `json:"task"`
	Completed bool   `json:"completed"`
	Commited  bool   `json:"commited"`
}

type TodoStorage struct {
	dict map[string]Todo
	// ključavnica za bralne in pisalne dostope do shrambe
	mu sync.RWMutex
}

var ErrorNotFound = errors.New("not found")

func NewTodoStorage() *TodoStorage {
	dict := make(map[string]Todo)
	return &TodoStorage{
		dict: dict,
	}
}

func (tds *TodoStorage) Get(todo *Todo, dict *map[string]Todo) error {
	tds.mu.RLock()
	defer tds.mu.RUnlock()
	if todo.Task == "" {
		for k, v := range tds.dict {
			(*dict)[k] = v
		}
		return nil
	} else {
		if val, ok := tds.dict[todo.Task]; ok {
			(*dict)[val.Task] = val
			return nil
		}
		return ErrorNotFound
	}
}

func (tds *TodoStorage) Put(todo *Todo) error { // , ret *struct{}
	tds.mu.Lock()
	defer tds.mu.Unlock()
	todo.Commited = false
	tds.dict[todo.Task] = *todo
	return nil

}

func (tds *TodoStorage) Commit(todo *Todo) error { // , ret *struct{}
	tds.mu.Lock()
	defer tds.mu.Unlock()
	if t, ok := tds.dict[todo.Task]; ok {
		t.Commited = true
		tds.dict[todo.Task] = t
		return nil
	}
	return ErrorNotFound

}

func main() {
	storage := NewTodoStorage()

	putTodo := Todo{"hej", true, true}
	storage.Put(&putTodo)

	todo := Todo{"", true, true}
	returnDict := map[string]Todo{}

	storage.Get(&todo, &returnDict)
	fmt.Println(returnDict)

	commitTodo := Todo{"hej", false, false}
	storage.Commit(&commitTodo)

	storage.Get(&todo, &returnDict)
	fmt.Println(returnDict)
}
