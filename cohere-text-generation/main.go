package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"

	"github.com/abhirockzz/amazon-bedrock-go-inference-params/cohere"
	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
)

const defaultRegion = "us-east-1"

const (
	cohereCommandModelID = "cohere.command-text-v14" //https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids-arns.html
)

const prompt1 = `Extract the band name from the contract:

This Music Recording Agreement ("Agreement") is made effective as of the 13 day of December, 2021 by and between Good Kid, a Toronto-based musical group (“Artist”) and Universal Music Group, a record label with license number 545345 (“Recording Label"). Artist and Recording Label may each be referred to in this Agreement individually as a "Party" and collectively as the "Parties." Work under this Agreement shall begin on March 15, 2022.`

const prompt = `Turn the following product feature into a list of benefits. Then, group this list of benefits into three types of benefits: Functional Benefits, Emotional Benefits, and Social Benefits.
"Product Feature:
Our app automatically transcribes your meetings. It uses state-of-the-art speech-to-text technology that works even in noisy backgrounds. Once the transcription is done, our app creates its summary and automatically emails it to the meeting attendees."`

func main() {

	region := os.Getenv("AWS_REGION")
	if region == "" {
		region = defaultRegion
	}

	cfg, err := config.LoadDefaultConfig(context.Background(), config.WithRegion(region))
	if err != nil {
		log.Fatal(err)
	}

	brc := bedrockruntime.NewFromConfig(cfg)

	payload := cohere.Request{
		Prompt:            prompt,
		Temperature:       0.40,
		P:                 0.75,
		K:                 0,
		MaxTokens:         200,
		ReturnLikelihoods: cohere.None,
	}

	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		log.Fatal(err)
	}

	output, err := brc.InvokeModel(context.Background(), &bedrockruntime.InvokeModelInput{
		Body:        payloadBytes,
		ModelId:     aws.String(cohereCommandModelID),
		ContentType: aws.String("application/json"),
		Accept:      aws.String("*/*"),
	})

	if err != nil {
		log.Fatal("failed to invoke model: ", err)
	}

	//log.Println("raw response ", string(output.Body))

	var resp cohere.Response

	err = json.Unmarshal(output.Body, &resp)

	if err != nil {
		log.Fatal("failed to unmarshal", err)
	}

	fmt.Println("response from LLM\n", resp.Generations[0].Text)

}
