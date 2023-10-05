package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrock"
)

const defaultRegion = "us-east-1"

func main() {

	region := os.Getenv("AWS_REGION")
	if region == "" {
		region = defaultRegion
	}

	cfg, err := config.LoadDefaultConfig(context.Background(), config.WithRegion(region))
	if err != nil {
		log.Fatal(err)
	}

	bc := bedrock.NewFromConfig(cfg)

	fms, err := bc.ListFoundationModels(context.Background(), &bedrock.ListFoundationModelsInput{
		//ByProvider: aws.String("Amazon"),
		//ByOutputModality: types.ModelModalityText,
	})

	if err != nil {
		fmt.Println("failed to list foundation models")
		log.Fatal(err)
	}

	for _, fm := range fms.ModelSummaries {
		fmt.Println(fmt.Sprintf("Name: %s | Provider: %s | Id: %s | Modality: %s", *fm.ModelName, *fm.ProviderName, *fm.ModelId, fm.OutputModalities))
	}

}
