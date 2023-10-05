package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"

	"github.com/abhirockzz/amazon-bedrock-go-inference-params/claude"
	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
)

const defaultRegion = "us-east-1"

const (
	claudePromptFormat = "\n\nHuman:%s\n\nAssistant:"
	claudeV2ModelID    = "anthropic.claude-v2" //https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids-arns.html
)

const prompt = `<paragraph> 
"In 1758, the Swedish botanist and zoologist Carl Linnaeus published in his Systema Naturae, the two-word naming of species (binomial nomenclature). Canis is the Latin word meaning "dog", and under this genus, he listed the domestic dog, the wolf, and the golden jackal."
</paragraph>

Please rewrite the above paragraph to make it understandable to a 5th grader.

Please output your rewrite in <rewrite></rewrite> tags.`

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

	payload := claude.Request{
		Prompt:            fmt.Sprintf(claudePromptFormat, prompt),
		MaxTokensToSample: 2048,
		Temperature:       0.5,
		TopK:              250,
		TopP:              1,
	}

	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		log.Fatal(err)
	}

	output, err := brc.InvokeModel(context.Background(), &bedrockruntime.InvokeModelInput{
		Body:        payloadBytes,
		ModelId:     aws.String(claudeV2ModelID),
		ContentType: aws.String("application/json"),
	})

	if err != nil {
		log.Fatal("failed to invoke model: ", err)
	}

	var resp claude.Response

	err = json.Unmarshal(output.Body, &resp)

	if err != nil {
		log.Fatal("failed to unmarshal", err)
	}

	fmt.Println("response from LLM\n", resp.Completion)

}
