package main
import (
    "bufio"
    // "time"
    "strings"
    "os"
    "fmt"
    "strconv"
    "sort"
)

func between(item int, low int, high int) bool {
    return (low <= item && item <= high)
}

func readEdgeList(filename string) []int {
    var field1 int
    var ctr int = 0
    var payload []int
    f, e := os.Open(filename)
    if e != nil {
        panic(e)
    }
    
    defer f.Close()

    s := bufio.NewScanner(f)
    for s.Scan() {
        line := strings.Fields(s.Text())
        // fmt.Println(line)
        field1,_ = strconv.Atoi(line[0])
        payload = append(payload,field1)
        ctr += 1
    }
    return payload
}

func readTADBoundaries(filename string, nodes []int) ([]int, []int, int) {
    var field2 int
    var ctr int = 0
    var payload []int
    g, e := os.Open(filename)
    if e != nil {
        panic(e)
    }
    
    defer g.Close()

    s := bufio.NewScanner(g)
    for s.Scan() {
        line := strings.Fields(s.Text())
        // fmt.Println(line)
        field2,_ = strconv.Atoi(line[1])
        payload = append(payload,field2)
        ctr += 1
    }
    sort.Ints(payload)

    // var community_ids []int
    community_ids := make([]int, len(nodes))
    var ptr1 int = 0
    var ptr2 int = 0
    for i := 0; i<len(payload); i++ {
        ptr1 = ptr2
        ptr2 = payload[i]
        for j := 1; j<len(nodes); j++ {
            // fmt.Println(nodes[j], ptr1, ptr2)
            switch{
                case between(nodes[j], ptr1, ptr2) == true:
                    // community_ids = append(community_ids, j)
                    community_ids[j] = i
                    continue
                case nodes[j] >= payload[len(payload)-1]:
                    // community_ids = append(community_ids, len(payload))
                    community_ids[j] = len(payload)
                    continue
                case nodes[j] <= payload[0]:
                    community_ids[j] = 0
                    // community_ids = append(community_ids, 0)
                    continue
                default:
                    continue
            }
        }
    }
    return payload, community_ids, ctr
}

func writeTADCenters(tadbank []int, communityIds []int, nodes []int) (map[int]int, []int) {
    var tadDict map[int]int
    tadDict = make(map[int]int)
    for i := 0; i<len(nodes); i++ {
        tadDict[nodes[i]] = communityIds[i]
    }

    var centers []int
    for i := 0; i<len(tadbank)+1; i++ {
        var templist []int
        for key, value := range tadDict {
            if value == i{
                templist = append(templist,key)
            }

        }
        sort.Ints(templist)
        if len(templist)>0 {
            divisor := len(templist)/2
            if divisor==0 {
                centers = append(centers,templist[0])
                continue
            }
            centers = append(centers,templist[divisor])
        }
    }
    return tadDict, centers
}

func main() {
    // start := time.Now()
    nodes := readEdgeList(os.Args[1])
    tadbank, communityids, _ := readTADBoundaries(os.Args[2],nodes)
    taddict, centers := writeTADCenters(tadbank, communityids, nodes)

    for i := 0; i<len(nodes); i++ {
        // Block{
        //     Try:
        //         centered := centers[taddict[nodes[i]]]
        //     Catch:
        // }.Do()
        centered := centers[taddict[nodes[i]]]
        fmt.Printf("%d\t%d\n", nodes[i], centered)
    }


    // elapsed := time.Since(start)
    // fmt.Printf("file operation took %s\n", elapsed)
}
