package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/abhirockzz/amazon-bedrock-go-inference-params/claude"
	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
)

const defaultRegion = "us-east-1"

var brc *bedrockruntime.Client

func init() {

	region := os.Getenv("AWS_REGION")
	if region == "" {
		region = defaultRegion
	}

	cfg, err := config.LoadDefaultConfig(context.Background(), config.WithRegion(region))
	if err != nil {
		log.Fatal(err)
	}

	brc = bedrockruntime.NewFromConfig(cfg)
}

var verbose *bool

func main() {
	verbose = flag.Bool("verbose", false, "setting to true will log messages being exchanged with LLM")
	flag.Parse()

	reader := bufio.NewReader(os.Stdin)

	var chatHistory string

	for {
		fmt.Print("\nEnter your message: ")
		input, _ := reader.ReadString('\n')
		input = strings.TrimSpace(input)

		msg := chatHistory + fmt.Sprintf(claudePromptFormat, input)

		response, err := send(msg)

		if err != nil {
			log.Fatal(err)
		}

		chatHistory = msg + response

		fmt.Println("\n--- Response ---")
		fmt.Println(response)
	}
}

const claudePromptFormat = "\n\nHuman: %s\n\nAssistant:"

func send(msg string) (string, error) {

	if *verbose {
		fmt.Println("[sending message]", msg)
	}

	payload := claude.Request{Prompt: msg, MaxTokensToSample: 2048}

	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return "", err
	}

	output, err := brc.InvokeModel(context.Background(), &bedrockruntime.InvokeModelInput{
		Body:        payloadBytes,
		ModelId:     aws.String("anthropic.claude-v2"),
		ContentType: aws.String("application/json"),
	})

	if err != nil {
		return "", err
	}

	var resp claude.Response

	err = json.Unmarshal(output.Body, &resp)

	if err != nil {
		return "", err
	}

	return resp.Completion, nil
}
