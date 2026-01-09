package portutil

import (
	"testing"
)

func TestParseListenAddr(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		want    string
		wantErr bool
	}{
		{
			name:  "empty uses default",
			input: "",
			want:  ":18051",
		},
		{
			name:  "port number only",
			input: "50052",
			want:  ":50052",
		},
		{
			name:  "colon prefix format",
			input: ":50052",
			want:  ":50052",
		},
		{
			name:  "localhost with port",
			input: "localhost:50052",
			want:  "localhost:50052",
		},
		{
			name:  "0.0.0.0 with port",
			input: "0.0.0.0:50052",
			want:  "0.0.0.0:50052",
		},
		{
			name:  "127.0.0.1 with port",
			input: "127.0.0.1:8080",
			want:  "127.0.0.1:8080",
		},
		{
			name:    "invalid port number",
			input:   "abc",
			wantErr: true,
		},
		{
			name:    "port too low",
			input:   "0",
			wantErr: true,
		},
		{
			name:    "port too high",
			input:   "65536",
			wantErr: true,
		},
		{
			name:    "invalid port in address",
			input:   ":abc",
			wantErr: true,
		},
		{
			name:    "too many colons",
			input:   "a:b:c",
			wantErr: true,
		},
		{
			name:  "min valid port",
			input: "1",
			want:  ":1",
		},
		{
			name:  "max valid port",
			input: "65535",
			want:  ":65535",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ParseListenAddr(tt.input)
			if (err != nil) != tt.wantErr {
				t.Errorf("ParseListenAddr(%q) error = %v, wantErr %v", tt.input, err, tt.wantErr)
				return
			}
			if !tt.wantErr && got != tt.want {
				t.Errorf("ParseListenAddr(%q) = %q, want %q", tt.input, got, tt.want)
			}
		})
	}
}

func TestResolvePort(t *testing.T) {
	tests := []struct {
		name       string
		portFlag   int
		listenFlag string
		want       string
		wantErr    bool
	}{
		{
			name:       "both empty uses default",
			portFlag:   0,
			listenFlag: "",
			want:       ":18051",
		},
		{
			name:       "port flag only",
			portFlag:   50052,
			listenFlag: "",
			want:       ":50052",
		},
		{
			name:       "listen flag only with colon",
			portFlag:   0,
			listenFlag: ":50053",
			want:       ":50053",
		},
		{
			name:       "listen flag only without colon",
			portFlag:   0,
			listenFlag: "50053",
			want:       ":50053",
		},
		{
			name:       "port flag takes precedence",
			portFlag:   50052,
			listenFlag: ":50053",
			want:       ":50052",
		},
		{
			name:       "port flag takes precedence over host:port",
			portFlag:   50052,
			listenFlag: "localhost:50053",
			want:       ":50052",
		},
		{
			name:       "listen flag with host",
			portFlag:   0,
			listenFlag: "0.0.0.0:50054",
			want:       "0.0.0.0:50054",
		},
		{
			name:       "port flag out of range low",
			portFlag:   0,
			listenFlag: "",
			want:       ":18051",
		},
		{
			name:       "port flag negative treated as unset",
			portFlag:   -1,
			listenFlag: "",
			wantErr:    true,
		},
		{
			name:       "port flag too high",
			portFlag:   65536,
			listenFlag: "",
			wantErr:    true,
		},
		{
			name:       "invalid listen flag",
			portFlag:   0,
			listenFlag: "invalid",
			wantErr:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ResolvePort(tt.portFlag, tt.listenFlag)
			if (err != nil) != tt.wantErr {
				t.Errorf("ResolvePort(%d, %q) error = %v, wantErr %v", tt.portFlag, tt.listenFlag, err, tt.wantErr)
				return
			}
			if !tt.wantErr && got != tt.want {
				t.Errorf("ResolvePort(%d, %q) = %q, want %q", tt.portFlag, tt.listenFlag, got, tt.want)
			}
		})
	}
}
