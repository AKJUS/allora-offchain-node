package apiadapter

import (
	"allora_offchain_node/lib"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"

	alloraMath "github.com/allora-network/allora-chain/math"
	"github.com/rs/zerolog/log"
)

type AlloraAdapter struct {
	name string
}

func (a *AlloraAdapter) Name() string {
	return a.name
}

func replacePlaceholders(urlTemplate string, params map[string]string) string {
	for key, value := range params {
		placeholder := fmt.Sprintf("{%s}", key)
		urlTemplate = strings.ReplaceAll(urlTemplate, placeholder, value)
	}
	return urlTemplate
}

// Replace placeholders and also the blockheheight
func replaceExtendedPlaceholders(urlTemplate string, params map[string]string, blockHeight int64, topicId uint64) string {
	// Create a map of default parameters
	blockHeightAsString := strconv.FormatInt(blockHeight, 10)
	topicIdAsString := strconv.FormatUint(topicId, 10)
	defaultParams := map[string]string{
		"BlockHeight": blockHeightAsString,
		"TopicId":     topicIdAsString,
	}
	urlTemplate = replacePlaceholders(urlTemplate, defaultParams)
	urlTemplate = replacePlaceholders(urlTemplate, params)
	return urlTemplate
}

func requestEndpoint(url string) (string, error) {
	// make request to url
	resp, err := http.Get(url) // nolint: gosec
	if err != nil {
		return "", fmt.Errorf("failed to make request to %s: %w", url, err)
	}
	defer resp.Body.Close()

	// Check if the response status is OK
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("received non-OK HTTP status %d", resp.StatusCode)
	}

	// Read the response body
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response body: %w", err)
	}

	log.Debug().Bytes("body", body).Msg("Requested endpoint")
	// convert bytes to string
	return string(body), nil
}

// parseJSONToNodeValues parses the incoming JSON string and returns a slice of NodeValue.
func parseJSONToNodeValues(jsonStr string) ([]lib.NodeValue, error) {
	// Define a map to hold the parsed JSON data.
	var data map[string][]float64

	// Parse the JSON string into the map.
	err := json.Unmarshal([]byte(jsonStr), &data)
	if err != nil {
		return nil, err
	}

	// Create a slice to hold the NodeValues.
	var nodeValues []lib.NodeValue

	// Iterate over the map to create NodeValue structs.
	for worker, values := range data {
		if len(values) > 0 {
			// Only pick the first value in the list.
			nodeValue := lib.NodeValue{
				Worker: worker,
				Value:  fmt.Sprintf("%f", values[0]),
			}
			nodeValues = append(nodeValues, nodeValue)
		}
	}

	return nodeValues, nil
}

// Expects an inference as a string scalar value
func (a *AlloraAdapter) CalcInference(node lib.WorkerConfig, blockHeight int64) (string, error) {
	log := log.With().Str("actorType", "worker").Uint64("topicId", node.TopicId).Logger()

	urlTemplate := node.Parameters["InferenceEndpoint"]
	url := replaceExtendedPlaceholders(urlTemplate, node.Parameters, blockHeight, node.TopicId)
	log.Debug().Str("url", url).Msg("Inference endpoint")
	return requestEndpoint(url)
}

// Expects forecast as a json array of NodeValue
func (a *AlloraAdapter) CalcForecast(node lib.WorkerConfig, blockHeight int64) ([]lib.NodeValue, error) {
	log := log.With().Str("actorType", "worker").Uint64("topicId", node.TopicId).Logger()

	urlTemplate := node.Parameters["ForecastEndpoint"]
	url := replaceExtendedPlaceholders(urlTemplate, node.Parameters, blockHeight, node.TopicId)
	log.Debug().Str("url", url).Msg("Forecasts endpoint")

	forecastsAsJsonString, err := requestEndpoint(url)
	if err != nil {
		log.Error().Err(err).Msg("Failed to get forecasts")
		return []lib.NodeValue{}, err
	}

	// parse json forecasts into a slice of NodeValue
	nodeValues, err := parseJSONToNodeValues(forecastsAsJsonString)
	if err != nil {
		log.Error().Err(err).Msg("Error transforming forecasts")
		return []lib.NodeValue{}, err
	}
	return nodeValues, nil
}

func (a *AlloraAdapter) GroundTruth(node lib.ReputerConfig, blockHeight int64) (lib.Truth, error) {
	log := log.With().Str("actorType", "reputer").Uint64("topicId", node.TopicId).Logger()

	urlTemplate := node.GroundTruthParameters["GroundTruthEndpoint"]
	url := replaceExtendedPlaceholders(urlTemplate, node.GroundTruthParameters, blockHeight, node.TopicId)
	log.Debug().Str("url", url).Msg("Ground truth endpoint")
	groundTruth, err := requestEndpoint(url)
	if err != nil {
		log.Error().Err(err).Msg("Failed to get ground truth")
		return "", err
	}
	// Check conversion to decimal before handing it over
	groundTruthDec, err := alloraMath.NewDecFromString(groundTruth)
	if err != nil {
		groundTruthDec, err = alloraMath.NewDecFromString(sanitizeDecString(groundTruth))
		if err != nil {
			log.Error().Err(err).Msg("Failed to convert ground truth to decimal")
			return "", err
		}
	}
	log.Info().Str("url", url).Str("groundTruth", groundTruthDec.String()).Msg("Ground truth")
	return groundTruthDec.String(), nil
}

func (a *AlloraAdapter) LossFunction(node lib.ReputerConfig, groundTruth string, inferenceValue string, options map[string]string) (string, error) {
	log := log.With().Str("actorType", "reputer").Uint64("topicId", node.TopicId).Logger()

	url := node.LossFunctionParameters.LossFunctionService
	if url == "" {
		return "", fmt.Errorf("no loss function endpoint provided")
	}
	// Use /calculate endpoint of loss-functions service
	url = fmt.Sprintf("%s/calculate", url)
	log.Debug().Str("url", url).Msg("Loss function endpoint")

	// Prepare the request payload
	payload := map[string]interface{}{
		"y_true":  groundTruth,
		"y_pred":  inferenceValue,
		"options": options,
	}

	// Convert payload to JSON
	jsonPayload, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("failed to marshal payload: %w", err)
	}

	// Create a new POST request
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonPayload))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	// Send the request
	client := &http.Client{} // nolint: exhaustruct
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	// Check the response status
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("received non-OK HTTP status %d", resp.StatusCode)
	}

	// Read and parse the response
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response body: %w", err)
	}

	var result struct {
		Loss string `json:"loss"`
	}
	if err := json.Unmarshal(body, &result); err != nil {
		return "", fmt.Errorf("failed to parse response: %w", err)
	}

	log.Debug().Str("url", url).Str("Loss", result.Loss).Msg("Calculated loss value from external endpoint")
	return result.Loss, nil
}

func (a *AlloraAdapter) IsLossFunctionNeverNegative(node lib.ReputerConfig, options map[string]string) (bool, error) {
	log := log.With().Str("actorType", "reputer").Uint64("topicId", node.TopicId).Logger()
	url := node.LossFunctionParameters.LossFunctionService
	if url == "" {
		return false, fmt.Errorf("no loss function endpoint provided")
	}
	// Use /is_never_negative endpoint of loss-functions service
	url = fmt.Sprintf("%s/is_never_negative", url)
	log.Debug().Str("url", url).Msg("Checking if loss function is never negative - endpoint")

	// Prepare the request payload
	payload := map[string]interface{}{
		"options": options,
	}

	// Convert payload to JSON
	jsonPayload, err := json.Marshal(payload)
	if err != nil {
		return false, fmt.Errorf("failed to marshal payload: %w", err)
	}

	// Create a new POST request
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonPayload))
	if err != nil {
		return false, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	// Send the request
	client := &http.Client{} // nolint: exhaustruct
	resp, err := client.Do(req)
	if err != nil {
		return false, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	// Check the response status
	if resp.StatusCode != http.StatusOK {
		return false, fmt.Errorf("received non-OK HTTP status %d", resp.StatusCode)
	}

	// Read and parse the response
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return false, fmt.Errorf("failed to read response body: %w", err)
	}

	var result struct {
		IsNeverNegative bool `json:"is_never_negative"`
	}
	if err := json.Unmarshal(body, &result); err != nil {
		return false, fmt.Errorf("failed to parse response: %w", err)
	}

	log.Debug().Str("url", url).Interface("options", options).Bool("IsNeverNegative", result.IsNeverNegative).Msg("Checked if loss function is never negative")
	return result.IsNeverNegative, nil
}

func (a *AlloraAdapter) CanInfer() bool {
	return true
}

func (a *AlloraAdapter) CanForecast() bool {
	return true
}

func (a *AlloraAdapter) CanSourceGroundTruthAndComputeLoss() bool {
	return true
}

func NewAlloraAdapter() *AlloraAdapter {
	return &AlloraAdapter{
		name: "apiAdapter",
	}
}

func sanitizeDecString(input string) string {
	// Remove any double quotes
	input = strings.ReplaceAll(input, "\"", "")

	// Remove any leading/trailing whitespace
	input = strings.TrimSpace(input)

	// Remove any commas (often used as thousand separators)
	input = strings.ReplaceAll(input, ",", "")

	// Ensure only one decimal point
	parts := strings.Split(input, ".")
	if len(parts) > 2 {
		input = parts[0] + "." + strings.Join(parts[1:], "")
	}

	return input
}
