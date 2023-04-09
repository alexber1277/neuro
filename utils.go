package neuro

import (
	"encoding/json"
	"errors"
	"io/ioutil"
	"log"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"time"
)

type KLine struct {
	PointTime     time.Time `json:"point_time"`
	PointTimeStr  string    `json:"point_time_str"`
	StartPrice    float64   `json:"start_price"`
	EndPrice      float64   `json:"end_price"`
	Volume        float64   `json:"volume"`
	VolumePers    float64   `json:"volume_pers"`
	Percent       float64   `json:"percent"`
	TradesPers    float64   `json:"trades_pers"`
	PriceArr      []float64 `json:"price_arr"`
	PercentArr    []float64 `json:"percent_arr"`
	TradesPersArr []float64 `json:"trades_pers_arr"`
	VolumePersArr []float64 `json:"volume_pers_arr"`
	NextPercent   float64   `json:"next_percent"`
	NextVolume    float64   `json:"next_volume"`
	NextTrades    float64   `json:"next_trades"`
	BuySaleStatus int       `json:"buy_sale_satus"`
	Trades        int       `json:"trades"`
	Index         int       `json:"index"`
	St            bool      `json:"st"`
}

const (
	urlBinance = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1h&limit=1000&startTime={$START}"
)

func GetData(period, last int, filename string) []DataTeach {
	var dataList []*KLine
	if bts, err := ioutil.ReadFile(filename); err != nil {
		for i := 1; i <= period; i++ {
			st := time.Now().Add(time.Hour * (1000 * time.Duration(i)) * -1)
			var list [][]interface{}
			if err := request(formUrl(st), &list); err != nil {
				log.Fatal("error request:", err)
			}
			klines, err := formatData(list)
			if err != nil {
				log.Fatal("error format:", err)
			}
			dataList = append(dataList, klines...)
			log.Println("iterate parse: ", i)
		}
		sorted(dataList)
		bts, err := json.Marshal(dataList)
		if err != nil {
			log.Fatal(err)
		}
		if err := ioutil.WriteFile(filename, bts, 0644); err != nil {
			log.Println(err)
		}
	} else {
		json.Unmarshal(bts, &dataList)
	}

	setPercent(dataList)
	setArrPriv(last, dataList)
	dataList = dataList[last+1:]

	return formedData(dataList)
}

func formedData(klines []*KLine) []DataTeach {
	var data []DataTeach
	for _, el := range klines {
		dt := DataTeach{}
		dt.Inputs = append(dt.Inputs, el.PercentArr...)
		dt.Inputs = append(dt.Inputs, el.TradesPersArr...)
		dt.Inputs = append(dt.Inputs, el.VolumePersArr...)
		dt.Price = el.EndPrice
		data = append(data, dt)
	}
	return data
}

func setPercent(klines []*KLine) {
	for i, el := range klines {
		if i == 0 {
			continue
		}
		el.Percent = getDiff(el.EndPrice, klines[i-1].EndPrice)
		el.TradesPers = middler(getDiff(float64(el.Trades), float64(klines[i-1].Trades)))
		el.VolumePers = middler(getDiff(float64(el.Volume), float64(klines[i-1].Volume)))
	}
}

func CheckResp(fls []float64) bool {
	if fls[0] == 0 && fls[1] == 0 && fls[2] == 0 {
		return false
	}
	if fls[0] == 1 && fls[1] == 1 && fls[2] == 1 {
		return false
	}
	if fls[0] == 1 && fls[1] == 0 && fls[2] == 1 {
		return false
	}
	if fls[0] == 1 && fls[1] == 1 && fls[2] == 0 {
		return false
	}
	if fls[0] == 0 && fls[1] == 1 && fls[2] == 1 {
		return false
	}
	return true
}

func middler(x float64) float64 {
	return toFixed(x/100, 3)
}

func setArrPriv(periodList int, klines []*KLine) {
	for i, el := range klines {
		if i >= periodList {
			el.St = true
			el.PriceArr = nil
			el.PercentArr = nil
			el.TradesPersArr = nil
			el.VolumePersArr = nil
			for _, elp := range klines[i-periodList : i] {
				el.PriceArr = append(el.PriceArr, elp.EndPrice)
				el.PercentArr = append(el.PercentArr, elp.Percent)
				el.TradesPersArr = append(el.TradesPersArr, elp.TradesPers)
				el.VolumePersArr = append(el.VolumePersArr, elp.VolumePers)
			}
		}
	}
}

func sorted(dataList []*KLine) {
	sort.Slice(dataList, func(i, j int) bool {
		return dataList[i].PointTime.Unix() < dataList[j].PointTime.Unix()
	})
}

func formUrl(start time.Time) string {
	return strings.Replace(urlBinance, "{$START}", strconv.FormatInt(start.UnixMilli(), 10), -1)
}

func request(str string, in interface{}) error {
	resp, er := http.Get(str)
	if er != nil {
		return er
	}
	bts, er := ioutil.ReadAll(resp.Body)
	if er != nil {
		return er
	}
	if err := json.Unmarshal(bts, &in); err != nil {
		return err
	}
	return nil
}

func formatData(ins [][]interface{}) ([]*KLine, error) {
	var list []*KLine
	if len(ins) == 0 {
		return list, errors.New("error empty response")
	}
	for _, el := range ins {
		kl := &KLine{}
		fl := el[0].(float64)
		kl.PointTime = time.UnixMilli(int64(fl))
		kl.PointTimeStr = kl.PointTime.Format("20060102150405")
		if elI, er := strconv.ParseFloat(el[1].(string), 64); er != nil {
			return list, er
		} else {
			kl.StartPrice = elI
		}
		if elI, er := strconv.ParseFloat(el[4].(string), 64); er != nil {
			return list, er
		} else {
			kl.EndPrice = elI
		}
		if elI, er := strconv.ParseFloat(el[5].(string), 64); er != nil {
			return list, er
		} else {
			kl.Volume = elI
		}
		kl.Trades = int(el[8].(float64))
		list = append(list, kl)
	}
	return list, nil
}
