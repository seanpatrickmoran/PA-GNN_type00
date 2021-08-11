package main

import (
    "bufio"
    "fmt"
    "log"
    "os"
    "strings"
    "strconv"
    "time"
)

//go run chromosomeSegmenter.go HiC/MicroC_ContactFreq.txt coordinate1 coordinate2 > outfile.txt
func main() {
    start := time.Now()
    f, e := os.Open(os.Args[1])
    if e != nil {
      panic(e)
    }
    defer f.Close()

    var field1 int
    var field2 int
    var llimit int
    var hlimit int

    llimit,_ = strconv.Atoi(os.Args[2])
    hlimit,_ = strconv.Atoi(os.Args[3])
    // log.Printf("%d\t%d",llimit,hlimit)
    s := bufio.NewScanner(f)
    for s.Scan() {
        hold := strings.Fields(s.Text())
       // log.Printf(strings.Fields(hold))
        field1,_ = strconv.Atoi(hold[0])
        field2,_ = strconv.Atoi(hold[1])

        if (field1>hlimit) {
            break
        }

        if(((llimit <= field1) && (field1 <= hlimit)) && ((llimit <= field2) && (field2 <= hlimit))){
            fmt.Printf("%d\t%d\t%s\n", field1, field2, hold[2])
        }
    }
    elapsed := time.Since(start)
    log.Printf("file operation took %s", elapsed)
}
