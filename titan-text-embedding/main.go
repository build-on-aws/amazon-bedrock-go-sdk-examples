package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"

	"github.com/abhirockzz/amazon-bedrock-go-inference-params/amazontitan/embedding"
	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
)

const defaultRegion = "us-east-1"

const (
	titanEmbeddingModelID = "amazon.titan-embed-text-v1" //https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids-arns.html
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

	input := os.Args[1]

	payload := embedding.Request{
		InputText: input,
	}

	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		log.Fatal(err)
	}

	output, err := brc.InvokeModel(context.Background(), &bedrockruntime.InvokeModelInput{
		Body:        payloadBytes,
		ModelId:     aws.String(titanEmbeddingModelID),
		ContentType: aws.String("application/json"),
	})

	if err != nil {
		log.Fatal("failed to invoke model: ", err)
	}

	var resp embedding.Response

	err = json.Unmarshal(output.Body, &resp)

	if err != nil {
		log.Fatal("failed to unmarshal", err)
	}

	fmt.Println("embedding vector from LLM\n", resp.Embedding)
	fmt.Println()

	fmt.Println("generated embedding for input -", input)
	fmt.Println("generated vector length -", len(resp.Embedding))
}
