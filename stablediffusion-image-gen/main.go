package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/abhirockzz/amazon-bedrock-go-inference-params/stabilityai"
	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
)

const defaultRegion = "us-east-1"

const (
	stableDiffusionXLModelID = "stability.stable-diffusion-xl-v0" //https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids-arns.html
)

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

	prompt := os.Args[1]
	fmt.Println("generating image based on prompt -", prompt)

	payload := stabilityai.Request{
		TextPrompts: []stabilityai.TextPrompt{{Text: prompt}},
		CfgScale:    10,
		Seed:        0,
		Steps:       50,
	}

	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		log.Fatal(err)
	}

	output, err := brc.InvokeModel(context.Background(), &bedrockruntime.InvokeModelInput{
		Body:        payloadBytes,
		ModelId:     aws.String(stableDiffusionXLModelID),
		ContentType: aws.String("application/json"),
	})

	if err != nil {
		log.Fatal("failed to invoke model: ", err)
	}

	var resp stabilityai.Response

	err = json.Unmarshal(output.Body, &resp)

	if err != nil {
		log.Fatal("failed to unmarshal", err)
	}

	decoded, err := resp.Artifacts[0].DecodeImage()

	if err != nil {
		log.Fatal("failed to decode base64 response", err)

	}

	outputFile := fmt.Sprintf("output-%d.jpg", time.Now().Unix())

	err = os.WriteFile(outputFile, decoded, 0644)
	if err != nil {
		fmt.Println("error writing to file:", err)
	}

	log.Println("image written to file", outputFile)

}
